#include <iostream>
#include <fstream>
#include <string>
#include <memory>

// #include "taichi_fixes.h"

#include "taichi/common/core.h"
// #include "taichi/common/types.h"
#include "taichi/common/zip.h"
#include "taichi/common/serialization.h"
#include "taichi/common/json_serde.h"
#include "taichi/ir/frontend_ir.h"
#include "taichi/rhi/arch.h"

using namespace taichi;
using namespace taichi::lang;
using namespace liong::json;

// Forward declarations
std::unique_ptr<Block> parse_block(const JsonValue& json_block);
Expr parse_expr(const JsonValue& json_expr);
std::unique_ptr<Stmt> parse_stmt(const JsonValue& json_stmt);

// Parse an expression from JSON
Expr parse_expr(const JsonValue& json_expr) {
    if (json_expr.is_null()) {
        return Expr();
    }

    std::string expr_type = (std::string)json_expr["type"];
    
    if (expr_type == "const") {
        // Parse const expression based on its data type
        std::string dt_type = (std::string)json_expr["dt"];
        if (dt_type == "i32") {
            return Expr(std::make_shared<ConstExpression>(
                (int)json_expr["value"]));
        } else if (dt_type == "f32") {
            return Expr(std::make_shared<ConstExpression>(
                (float)json_expr["value"]));
        } else if (dt_type == "i64") {
            return Expr(std::make_shared<ConstExpression>(
                (int64_t)json_expr["value"]));
        } else if (dt_type == "f64") {
            return Expr(std::make_shared<ConstExpression>(
                (double)json_expr["value"]));
        } else {
            std::cerr << "Unsupported const type: " << dt_type << std::endl;
        }
    } else if (expr_type == "id") {
        int id = (int)json_expr["id"];
        std::string name = "";
        if (!json_expr["name"].is_null()) {
            name = (std::string)json_expr["name"];
        }
        return Expr(std::make_shared<IdExpression>(Identifier(id, name)));
    } else if (expr_type == "unary") {
        UnaryOpType op_type;
        std::string op = (std::string)json_expr["op"];
        
        if (op == "neg") op_type = UnaryOpType::neg;
        else if (op == "not") op_type = UnaryOpType::logic_not;
        else if (op == "bit_not") op_type = UnaryOpType::bit_not;
        else if (op == "cast_float") op_type = UnaryOpType::cast_value;
        else if (op == "cast_int") op_type = UnaryOpType::cast_value;
        else {
            std::cerr << "Unsupported unary op: " << op << std::endl;
            op_type = UnaryOpType::undefined;
        }
        
        return Expr(std::make_shared<UnaryOpExpression>(
            op_type, parse_expr(json_expr["operand"])));
    } else if (expr_type == "binary") {
        BinaryOpType op_type;
        std::string op = (std::string)json_expr["op"];
        
        if (op == "add") op_type = BinaryOpType::add;
        else if (op == "sub") op_type = BinaryOpType::sub;
        else if (op == "mul") op_type = BinaryOpType::mul;
        else if (op == "div") op_type = BinaryOpType::div;
        else if (op == "mod") op_type = BinaryOpType::mod;
        else if (op == "cmp_lt") op_type = BinaryOpType::cmp_lt;
        else if (op == "cmp_le") op_type = BinaryOpType::cmp_le;
        else if (op == "cmp_gt") op_type = BinaryOpType::cmp_gt;
        else if (op == "cmp_ge") op_type = BinaryOpType::cmp_ge;
        else if (op == "cmp_eq") op_type = BinaryOpType::cmp_eq;
        else if (op == "cmp_ne") op_type = BinaryOpType::cmp_ne;
        else if (op == "bit_and") op_type = BinaryOpType::bit_and;
        else if (op == "bit_or") op_type = BinaryOpType::bit_or;
        else if (op == "bit_xor") op_type = BinaryOpType::bit_xor;
        else if (op == "logical_and") op_type = BinaryOpType::logical_and;
        else if (op == "logical_or") op_type = BinaryOpType::logical_or;
        else {
            std::cerr << "Unsupported binary op: " << op << std::endl;
            op_type = BinaryOpType::undefined;
        }
        
        return Expr(std::make_shared<BinaryOpExpression>(
            op_type, parse_expr(json_expr["lhs"]), parse_expr(json_expr["rhs"])));
    } else if (expr_type == "index") {
        Expr var = parse_expr(json_expr["var"]);
        
        // Parse indices
        const JsonValue& json_indices = json_expr["indices"];
        ExprGroup indices;
        for (const auto& index : json_indices.elems()) {
            indices.push_back(parse_expr(index));
        }
        
        return Expr(std::make_shared<IndexExpression>(var, indices));
    } else {
        std::cerr << "Unsupported expression type: " << expr_type << std::endl;
    }
    
    // Default empty expression if parsing failed
    return Expr();
}

// Parse a statement from JSON
std::unique_ptr<Stmt> parse_stmt(const JsonValue& json_stmt) {
    std::string stmt_type = (std::string)json_stmt["type"];
    
    if (stmt_type == "alloca") {
        int id = (int)json_stmt["id"];
        std::string name = "";
        if (!json_stmt["name"].is_null()) {
            name = (std::string)json_stmt["name"];
        }
        
        // Assume primitive type for now
        // A more complete implementation would need to handle tensor types
        Identifier ident(id, name);
        return std::make_unique<FrontendAllocaStmt>(ident, PrimitiveType::i32);
    } else if (stmt_type == "assign") {
        Expr lhs = parse_expr(json_stmt["lhs"]);
        Expr rhs = parse_expr(json_stmt["rhs"]);
        return std::make_unique<FrontendAssignStmt>(lhs, rhs);
    } else if (stmt_type == "if") {
        Expr condition = parse_expr(json_stmt["condition"]);
        auto stmt = std::make_unique<FrontendIfStmt>(condition, DebugInfo());
        
        if (!json_stmt["true_statements"].is_null()) {
            stmt->true_statements = parse_block(json_stmt["true_statements"]);
        } else {
            stmt->true_statements = std::make_unique<Block>();
        }
        
        if (!json_stmt["false_statements"].is_null()) {
            stmt->false_statements = parse_block(json_stmt["false_statements"]);
        } else {
            stmt->false_statements = std::make_unique<Block>();
        }
        
        return stmt;
    } else if (stmt_type == "for") {
        // Parse loop variable
        ExprGroup loop_vars;
        if (!json_stmt["loop_var"].is_null()) {
            loop_vars.push_back(parse_expr(json_stmt["loop_var"]));
        }
        
        // Check if it's a range for loop
        if (!json_stmt["begin"].is_null() && !json_stmt["end"].is_null()) {
            Expr begin = parse_expr(json_stmt["begin"]);
            Expr end = parse_expr(json_stmt["end"]);
            ForLoopConfig config;  // Default config
            
            auto stmt = std::make_unique<FrontendForStmt>(
                loop_vars[0], begin, end, Arch::x64, config);
            
            if (!json_stmt["body"].is_null()) {
                stmt->body = parse_block(json_stmt["body"]);
            } else {
                stmt->body = std::make_unique<Block>();
            }
            
            return stmt;
        }
        // Could add support for other for loop types (struct_for, etc.)
        
    } else if (stmt_type == "expr") {
        Expr val = parse_expr(json_stmt["value"]);
        return std::make_unique<FrontendExprStmt>(val);
    } else {
        std::cerr << "Unsupported statement type: " << stmt_type << std::endl;
    }
    
    // Return null if parsing failed
    return nullptr;
}

// Parse a block from JSON
std::unique_ptr<Block> parse_block(const JsonValue& json_block) {
    auto block = std::make_unique<Block>();
    
    if (json_block.is_arr()) {
        for (const auto& json_stmt : json_block.elems()) {
            auto stmt = parse_stmt(json_stmt);
            if (stmt) {
                block->statements.push_back(std::move(stmt));
            }
        }
    }
    
    return block;
}

// Build AST from JSON
std::unique_ptr<Block> build_ast_from_json(const JsonValue& json_ast) {
    // Extract the root block
    if (json_ast.is_obj() && !json_ast["body"].is_null()) {
        return parse_block(json_ast["body"]);
    }
    
    // Return empty block if no valid body found
    return std::make_unique<Block>();
}

int main(int argc, char* argv[]) {
    std::string filename = "_func_narrow_phase_c220_0_ast.json";
    
    // If a filename is provided as argument, use it instead
    if (argc > 1) {
        filename = argv[1];
    }
    
    // Read the JSON file
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return 1;
    }
    
    std::string json_content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    
    try {
        // Parse JSON
        JsonValue json_ast = liong::json::parse(json_content);
        std::cout << "Successfully parsed JSON" << std::endl;
        
        // Build AST from JSON
        auto ast_root = build_ast_from_json(json_ast);
        std::cout << "Successfully built AST with " 
                  << ast_root->statements.size() << " top-level statements" << std::endl;
        
        // Create a FrontendContext with the parsed AST
        FrontendContext frontend_ctx(Arch::x64, true);
        auto& builder = frontend_ctx.builder();
        
        // Here you could traverse the AST and add statements to the frontend_ctx
        // For demonstration, just printing the structure
        std::function<void(Block*, int)> print_ast = [&](Block* block, int depth) {
            std::string indent(depth * 2, ' ');
            
            for (const auto& stmt : block->statements) {
                if (auto if_stmt = dynamic_cast<FrontendIfStmt*>(stmt.get())) {
                    std::cout << indent << "If statement" << std::endl;
                    std::cout << indent << "  True branch:" << std::endl;
                    print_ast(if_stmt->true_statements.get(), depth + 2);
                    std::cout << indent << "  False branch:" << std::endl;
                    print_ast(if_stmt->false_statements.get(), depth + 2);
                } else if (auto for_stmt = dynamic_cast<FrontendForStmt*>(stmt.get())) {
                    std::cout << indent << "For statement" << std::endl;
                    std::cout << indent << "  Body:" << std::endl;
                    print_ast(for_stmt->body.get(), depth + 2);
                } else if (auto assign_stmt = dynamic_cast<FrontendAssignStmt*>(stmt.get())) {
                    std::cout << indent << "Assign statement" << std::endl;
                } else if (auto alloca_stmt = dynamic_cast<FrontendAllocaStmt*>(stmt.get())) {
                    std::cout << indent << "Alloca statement: " << alloca_stmt->ident.raw_name() << std::endl;
                } else {
                    std::cout << indent << "Other statement type" << std::endl;
                }
            }
        };
        
        print_ast(ast_root.get(), 0);
        
    } catch (const JsonException& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
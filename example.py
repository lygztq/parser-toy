from parsertoy import symbol, SymbolTable
import prettytable as pt

# with SymbolTable() as table:
#     for i in range(26):
#         curr_name = chr(ord('A') + i)
#         curr_symbol = symbol(name=curr_name)
#         for j in range(i):
#             curr_symbol.add_production( *[chr(ord('A') + k) for k in range(j + 1)] )

#     for sym_name, sym in table.symbols.items():
#         print(sym.check_valid())
#         print(sym)

# with SymbolTable() as table:
#     A = symbol("A", productions=[["A", "a"], ["b"]])
#     a = symbol("a")
#     b = symbol("b")

#     A.remove_left_recursions()
#     table.dump()

with SymbolTable() as table:
    E = symbol("Expr")
    T = symbol("Term")
    F = symbol("Factor")
    symbol("(")
    symbol(")")
    symbol("num")
    symbol("name")
    symbol("+")
    symbol("-")
    symbol("*")
    symbol("/")

    E.add_production("Expr", "+", "Term")
    E.add_production("Expr", "-", "Term")
    E.add_production("Term")

    T.add_production("Term", "*", "Factor")
    T.add_production("Term", "/", "Factor")
    T.add_production("Factor")

    F.add_production("(", "Expr", ")")
    F.add_production("num")
    F.add_production("name")

    table.dump()

    table.remove_left_recursions()
    table.left_factoring()

    table.dump()

    table.start_symbol = "Expr"
    assert table.is_valid()

    tb = pt.PrettyTable(["Production", "First+"])
    for k, v in table.build_first_plus_set().items():
        tb.add_row([k, set(sorted(list(v)))])
    tb.align["Production"] = "l"
    tb.align["First+"] = "l"
    print(tb)


# with SymbolTable() as table:
#     symbol("If")
#     symbol("Else")
#     E = symbol("expr")
#     symbol("bool")
#     symbol("Then")
#     S = symbol("Stmt")
#     S.add_production("If", "expr", "Then", "stmt", "else", "stmt")
#     S.add_production("If", "expr", "Then", "stmt")
#     E.add_production("bool")

#     table.remove_left_recursions()
#     table.left_factoring()
#     table.dump()


# with SymbolTable() as table:
#     F = symbol("Factor")
#     symbol("name")
#     symbol("[")
#     A = symbol("ArgList")
#     symbol("]")
#     symbol("(")
#     symbol(")")
#     symbol("Expr")
#     M = symbol("MoreArgs")
#     symbol(",")

#     F.add_production("name")
#     F.add_production("name", "[", "ArgList", "]")
#     F.add_production("name", "(", "ArgList", ")")
#     A.add_production("Expr", "MoreArgs")
#     M.add_production(",", "Expr", "MoreArgs")
#     M.add_production("epsilon")

#     table.remove_left_recursions()
#     table.left_factoring()
#     table.dump()


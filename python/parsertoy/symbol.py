from typing import Dict, List, Optional, Set
from collections import OrderedDict

SymbolSeq = List[str]
Production = List[str]
FirstSet = Dict[str, Set[str]]
FollowSet = Dict[str, Set[str]]

class Symbol(object):
    _disable_chars = ["="]
    EPSILON_NAME = "epsilon"
    EOF_NAME = "EOF"
    def __init__(self, name=None, productions=[]):
        """ Do NOT use this directly, use symbol instead """
        self.name : str = SymbolTable.current_table().get_anonymous_symbol_name() if name is None else name
        self.productions : List[Production] = productions.copy()

    def is_left_recursion(self, p: Production):
        return len(p) > 0 and self.name == p[0]

    def check_valid(self):
        table = SymbolTable.current_table()
        return all(map(lambda x: all(map(lambda s: table.has(s), x)), self.productions)) and table.has(self.name) and all(map(lambda x: x not in self.name, Symbol._disable_chars))

    def remove_duplicate(self):
        new_productions = []
        for p in self.productions:
            if p not in new_productions:
                new_productions.append(p)
        self.productions = new_productions

    def left_factoring(self):
        prefixs = {}
        div_char = Symbol._disable_chars[0]
        for p in self.productions:
            for i in range(len(p)):
                prefix = div_char.join(p[:i+1])
                if prefix in prefixs:
                    prefixs[prefix] += 1
                else:
                    prefixs[prefix] = 1

        max_prefix : Optional[str] = None
        max_len = 0
        for prefix, cnt in prefixs.items():
            p_list = prefix.split(div_char)
            if cnt < 2 or max_len > len(p_list):
                continue
            max_len = len(p_list)
            max_prefix = prefix

        if max_prefix is None:
            return False

        new_prod = []
        factor_remains = []
        for p in self.productions:
            if div_char.join(p).startswith(max_prefix):
                factor_remains.append(p[max_len:] if p[max_len:] else [self.EPSILON_NAME])
            else:
                new_prod.append(p)

        table = SymbolTable.current_table()
        new_name = self.name + '\''
        if new_name in table.symbols:
            new_name = None
        symbol(name=new_name, productions=factor_remains)
        new_prod.append(max_prefix.split(div_char) + [new_name])
        self.productions = new_prod
        return True

    def remove_left_recursions(self):
        table = SymbolTable.current_table()
        alphas = []
        betas = []
        for p in self.productions:
            if self.is_left_recursion(p):
                alphas.append(p[1:])
            else:
                betas.append(p)
        if not alphas:
            return
        new_name = self.name + '\''
        if new_name in table.symbols:
            new_name = None
        _ = symbol(name=new_name, productions=list(map(lambda p: p + [new_name], alphas)) + [[self.EPSILON_NAME]])
        self.productions = list(map(lambda p: p + [new_name], betas))

    @property
    def is_non_terminal(self) -> bool:
        return len(self.productions) > 0

    @property
    def is_terminal(self) -> bool:
        return len(self.productions) == 0 and not self.is_special

    @property
    def is_epsilon(self) -> bool:
        return self.name == self.EPSILON_NAME

    @property
    def is_eof(self) -> bool:
        return self.name == self.EOF_NAME

    @property
    def is_special(self) -> bool:
        return self.is_epsilon or self.is_eof

    @staticmethod
    def symbol_seq_first_value(sym_seq: SymbolSeq, first_set: FirstSet):
        first = set()
        to_the_end = True
        for sym in sym_seq:
            first |= first_set[sym]
            if Symbol.EPSILON_NAME not in first_set[sym]:
                to_the_end = False
                break
        if to_the_end:
            first.add(Symbol.EPSILON_NAME)
        return first

    def add_production(self, *symbols: Production):
        symbols = list(symbols)
        if not symbols:
            return
        if symbols not in self.productions:
            self.productions.append(symbols)

    def __str__(self) -> str:
        out_str = self.name
        if self.is_non_terminal:
            div = "\n{}| ".format(" " * (len(self.name) + 2))
            out_str += " -> "
            out_str += div.join(map(lambda x: " ".join(x), self.productions))
        return out_str
    __repr__ = __str__


def symbol(name=None, productions=[]):
    sym = Symbol(name, productions)
    table = SymbolTable.current_table()
    table.register_symbol(sym)
    return sym

class SymbolTable(object):
    _TABLE_STACK : List["SymbolTable"] = []
    def __init__(self) -> None:
        self.num_anonymous_symbol = 0
        self.symbols : OrderedDict[Symbol] = OrderedDict()
        self.symbols[Symbol.EPSILON_NAME] = Symbol(name=Symbol.EPSILON_NAME)
        self.symbols[Symbol.EOF_NAME] = Symbol(name=Symbol.EOF_NAME)
        self.start_symbol : Optional[str] = None

    def is_valid(self):
        valid = True
        for sym in self.symbols.values():
            valid = valid and sym.check_valid()
        valid = valid \
            and self.start_symbol is not None \
            and self.start_symbol in self.symbols
        return valid

    def has(self, name):
        return name in self.symbols

    def get_anonymous_symbol_name(self) -> str:
        name = "symbol_{}".format(self.num_anonymous_symbol)
        self.num_anonymous_symbol += 1
        return name

    def register_symbol(self, obj: Symbol):
        name = obj.name
        if name in self.symbols:
            raise ValueError("Find duplicate symbol {}".format(name))
        self.symbols[name] = obj

    def get_symbol(self, name: str) -> Optional[Symbol]:
        return self.symbols.get(name, None)

    def left_factoring(self):
        not_stop = True
        while not_stop:
            not_stop = False
            for sym in list(self.symbols.values()):
                not_stop |= sym.left_factoring()

    def remove_left_recursions(self):
        # build order
        order = {}
        non_terminals = []
        idx = 0
        for sym in self.symbols.values():
            if sym.is_non_terminal:
                order[sym.name] = idx
                idx += 1
                non_terminals.append(sym)

        for sym in non_terminals:
            for p in sym.productions:
                start_sym = self.get_symbol(p[0])
                if start_sym is None:
                    raise RuntimeError("Symbol {} is not defined".format(p[0]))
                remain = p[1:]
                if start_sym.is_non_terminal and order[start_sym.name] < order[sym.name]:
                    sym.productions.extend(list(map(lambda x: x + remain, start_sym.productions)))
            sym.remove_left_recursions()

    def build_first_set(self) -> FirstSet:
        first = {}
        for sym in self.symbols.values():
            if not sym.is_non_terminal:
                first[sym.name] = set([sym.name])
            else:
                first[sym.name] = set()

        not_change = False
        while not not_change:
            not_change = True
            for sym in self.symbols.values():
                for p in sym.productions:
                    rhs = set()
                    i = 0
                    if all(map(lambda x: x != Symbol.EPSILON_NAME, p)):
                        rhs = first[p[0]] - set([Symbol.EPSILON_NAME])
                        while Symbol.EPSILON_NAME in first[p[i]] and i < len(p) - 1:
                            rhs = rhs | (first[p[i + 1]] - set([Symbol.EPSILON_NAME]))
                            i += 1
                    if i == len(p) - 1 and Symbol.EPSILON_NAME in first[p[-1]]:
                        rhs = rhs | set([Symbol.EPSILON_NAME])
                    if rhs:
                        not_change = not_change and ((first[sym.name] | rhs) == first[sym.name])
                        first[sym.name] = first[sym.name] | rhs
        return first

    def build_follow_set(self, first=None) -> FollowSet:
        if first is None:
            first = self.build_first_set()

        follow = {}
        for sym in self.symbols.values():
            if sym.is_non_terminal:
                follow[sym.name] = set()
        follow[self.start_symbol] = set([Symbol.EOF_NAME])

        not_change = False
        while not not_change:
            not_change = True
            for sym in self.symbols.values():
                for p in sym.productions:
                    trailer = follow[sym.name]
                    for beta_name in reversed(p):
                        beta = self.symbols[beta_name]
                        if beta.is_non_terminal:
                            not_change = not_change and (follow[beta_name] == follow[beta_name] | trailer)
                            follow[beta_name] = follow[beta_name] | trailer
                            if Symbol.EPSILON_NAME in first[beta_name]: # if this can be empty
                                trailer = trailer | (first[beta_name] - set([Symbol.EPSILON_NAME]))
                            else:
                                trailer = first[beta_name]
                        else:
                            trailer = first[beta_name]
        return follow

    def build_first_plus_set(self, first=None, follow=None):
        if first is None:
            first = self.build_first_set()
            follow = self.build_follow_set(first)
        if follow is None:
            follow = self.build_follow_set(first)

        first_plus = {}
        for sym in self.symbols.values():
            for p in sym.productions:
                key = "{} -> {}".format(sym.name, " ".join(p))
                first_right_seq = Symbol.symbol_seq_first_value(p, first)
                if Symbol.EPSILON_NAME in first_right_seq:
                    value = first_right_seq | follow[sym.name]
                else:
                    value = first_right_seq
                first_plus[key] = value
        return first_plus

    @staticmethod
    def current_table():
        if not SymbolTable._TABLE_STACK:
            SymbolTable._TABLE_STACK.append(SymbolTable())
        return SymbolTable._TABLE_STACK[-1]

    @staticmethod
    def insert_table(table: "SymbolTable"):
        SymbolTable._TABLE_STACK.append(table)

    @staticmethod
    def pop_table():
        SymbolTable._TABLE_STACK.pop()

    def __enter__(self) -> "SymbolTable":
        SymbolTable.insert_table(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        SymbolTable.pop_table()

    def dump(self):
        non_terminals = []
        terminals = []
        for sym in self.symbols.values():
            if not sym.is_non_terminal:
                terminals.append(sym)
            else:
                non_terminals.append(sym)
        print("Non terminals and special:\n---")
        for sym in non_terminals:
            print(sym)
        print("\nTerminals:\n---")
        for sym in terminals:
            print(sym)

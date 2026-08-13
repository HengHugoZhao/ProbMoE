LITERAL, DECOMPOSISTION, TRUE = 0,1,2

class Node:
    
    node_id = 1
    def __init__(self, elements=None, type=DECOMPOSISTION):
        self.elements = elements
        self.id = Node.node_id
        self.type = type
        self._bit = False
        self.data = None
        
        if self.type == LITERAL:
            self.literal = elements

        Node.node_id += 1    

        
    def __repr__(self):
        return f"Node(id={self.id})"
    
    def is_decomposition(self):
        return self.type == DECOMPOSISTION
    
    def is_literal(self):
        return self.type == LITERAL
    
    def is_true(self):
        return self.type == TRUE
    
    def clear_bits(self, clear_data=False):
        if self._bit is False:
            return
        
        self._bit = False
        if clear_data: self.data = None
        if self.is_decomposition():
            for p, s in self.elements:
                p.clear_bits(clear_data=clear_data)
                s.clear_bits(clear_data=clear_data)
    
    def positive_iter(self, first_call=True, clear_data=False):
        if self._bit: return
        self._bit = True
        
        if self.is_decomposition():
            for p, s in self.elements:
                for node in p.positive_iter(first_call=False, clear_data=clear_data): yield node
                for node in s.positive_iter(first_call=False, clear_data=clear_data): yield node
        yield self
        
        if first_call:
            self.clear_bits(clear_data=clear_data)
        
        

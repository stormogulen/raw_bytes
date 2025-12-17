// Keep everything from before, but modify these parts:

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Opcode {
    Nop = 0,
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
    MovImm = 5,
    MovReg = 6,
    Kill = 7,
    
    // Vector operations (work on register pairs)
    Vec2Add = 8,
    Vec2Sub = 9,
    Vec2Mul = 10,  // Multiply vector by scalar
    Vec2Length = 11,
    Vec2Normalize = 12,
    
    // Math functions
    Sin = 13,
    Cos = 14,
    Sqrt = 15,

    // Control flow 
    Jump = 16,          // Unconditional jump
    JumpIfZero = 17,    // Jump if register == 0
    JumpIfNotZero = 18, // Jump if register != 0
    JumpIfNeg = 19,     // Jump if register < 0
    JumpIfPos = 20,     // Jump if register > 0
    
    // Comparisons 
    CmpLt = 21,         // ra = (ra < rb) ? 1.0 : 0.0
    CmpGt = 22,         // ra = (ra > rb) ? 1.0 : 0.0
    CmpEq = 23,         // ra = (ra == rb) ? 1.0 : 0.0
}

impl Opcode {
    fn from_u8(value: u8) -> Result<Self, String> {
        match value {
            0 => Ok(Opcode::Nop),
            1 => Ok(Opcode::Add),
            2 => Ok(Opcode::Sub),
            3 => Ok(Opcode::Mul),
            4 => Ok(Opcode::Div),
            5 => Ok(Opcode::MovImm),
            6 => Ok(Opcode::MovReg),
            7 => Ok(Opcode::Kill),
            8 => Ok(Opcode::Vec2Add),
            9 => Ok(Opcode::Vec2Sub),
            10 => Ok(Opcode::Vec2Mul),
            11 => Ok(Opcode::Vec2Length),
            12 => Ok(Opcode::Vec2Normalize),
            13 => Ok(Opcode::Sin),
            14 => Ok(Opcode::Cos),
            15 => Ok(Opcode::Sqrt),
            16 => Ok(Opcode::Jump),
            17 => Ok(Opcode::JumpIfZero),
            18 => Ok(Opcode::JumpIfNotZero),
            19 => Ok(Opcode::JumpIfNeg),
            20 => Ok(Opcode::JumpIfPos),
            21 => Ok(Opcode::CmpLt),
            22 => Ok(Opcode::CmpGt),
            23 => Ok(Opcode::CmpEq),
            _ => Err(format!("Invalid opcode: {}", value)),
        }
    }
}

// Add this at the top with other structs

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Instruction(String),
    Register(u8),
    VectorPair(u8),
    Number(f32),
    Label(String),
    LabelRef(String),
    Comma,
    Colon,
    Newline,
}

struct Lexer {
    input: String,
    pos: usize,
}

impl Lexer {
    fn new(input: String) -> Self {
        Self { input, pos: 0 }
    }
    
    fn tokenize(&mut self) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::new();
        
        while self.pos < self.input.len() {
            self.skip_whitespace_inline();
            
            if self.pos >= self.input.len() {
                break;
            }
            
            let ch = self.current_char();
            
            match ch {
                ';' => {
                    self.skip_line();
                    continue;
                }
                ',' => {
                    tokens.push(Token::Comma);
                    self.advance();
                }
                ':' => {
                    tokens.push(Token::Colon);
                    self.advance();
                }
                '\n' => {
                    tokens.push(Token::Newline);
                    self.advance();
                }
                '@' => {
                    self.advance();
                    let name = self.read_identifier();
                    tokens.push(Token::LabelRef(name));
                }
                _ if ch.is_alphabetic() || ch == '_' => {
                    let ident = self.read_identifier();
                    
                    // Check for register (r0-r7)
                    if ident.starts_with('r') && ident.len() == 2 {
                        if let Ok(num) = ident[1..].parse::<u8>() {
                            if num < 8 {
                                tokens.push(Token::Register(num));
                                continue;
                            }
                        }
                    }
                    
                    // Check for vector pair (v0-v3)
                    if ident.starts_with('v') && ident.len() == 2 {
                        if let Ok(num) = ident[1..].parse::<u8>() {
                            if num < 4 {
                                tokens.push(Token::VectorPair(num));
                                continue;
                            }
                        }
                    }
                    
                    // Check if next char is colon (label definition)
                    self.skip_whitespace_inline();
                    if self.pos < self.input.len() && self.current_char() == ':' {
                        tokens.push(Token::Label(ident));
                    } else {
                        tokens.push(Token::Instruction(ident));
                    }
                }
                _ if ch.is_numeric() || ch == '-' || ch == '.' => {
                    let num = self.read_number()?;
                    tokens.push(Token::Number(num));
                }
                _ => {
                    return Err(format!("Unexpected character: '{}'", ch));
                }
            }
        }
        
        Ok(tokens)
    }
    
    fn current_char(&self) -> char {
        self.input.chars().nth(self.pos).unwrap()
    }
    
    fn advance(&mut self) {
        self.pos += 1;
    }
    
    fn skip_whitespace_inline(&mut self) {
        while self.pos < self.input.len() {
            let ch = self.current_char();
            if ch == ' ' || ch == '\t' || ch == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }
    
    fn skip_line(&mut self) {
        while self.pos < self.input.len() && self.current_char() != '\n' {
            self.advance();
        }
    }
    
    fn read_identifier(&mut self) -> String {
        let start = self.pos;
        while self.pos < self.input.len() {
            let ch = self.current_char();
            if ch.is_alphanumeric() || ch == '_' {
                self.advance();
            } else {
                break;
            }
        }
        self.input[start..self.pos].to_string()
    }
    
    fn read_number(&mut self) -> Result<f32, String> {
        let start = self.pos;
        
        if self.current_char() == '-' {
            self.advance();
        }
        
        let mut has_dot = false;
        while self.pos < self.input.len() {
            let ch = self.current_char();
            if ch.is_numeric() {
                self.advance();
            } else if ch == '.' && !has_dot {
                has_dot = true;
                self.advance();
            } else {
                break;
            }
        }
        
        self.input[start..self.pos].parse::<f32>()
            .map_err(|e| format!("Invalid number: {}", e))
    }
}

struct Assembler {
    tokens: Vec<Token>,
    pos: usize,
    program: Program,
    labels: HashMap<String, usize>,
    fixups: Vec<(usize, String)>,
}

impl Assembler {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            program: Program::new(),
            labels: HashMap::new(),
            fixups: Vec::new(),
        }
    }
    
    fn assemble(&mut self) -> Result<Program, String> {
        // First pass: collect labels
        self.first_pass()?;
        
        // Second pass: emit code
        self.pos = 0;
        self.second_pass()?;
        
        // Third pass: fix up jumps
        self.fixup_jumps()?;
        
        Ok(self.program.clone())
    }
    
    fn first_pass(&mut self) -> Result<(), String> {
        let mut bit_pos = 0;
        
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos] {
                Token::Label(name) => {
                    self.labels.insert(name.clone(), bit_pos);
                    self.pos += 1;
                    if self.pos < self.tokens.len() && matches!(self.tokens[self.pos], Token::Colon) {
                        self.pos += 1;
                    }
                }
                Token::Instruction(name) => {
                    bit_pos += self.estimate_instruction_size(name)?;
                    self.skip_instruction();
                }
                Token::Newline => {
                    self.pos += 1;
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
        
        Ok(())
    }
    
    fn second_pass(&mut self) -> Result<(), String> {
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos].clone() {
                Token::Instruction(name) => {
                    self.emit_instruction(name)?;
                }
                Token::Label(_) => {
                    self.pos += 1;
                    if self.pos < self.tokens.len() && matches!(self.tokens[self.pos], Token::Colon) {
                        self.pos += 1;
                    }
                }
                Token::Newline => {
                    self.pos += 1;
                }
                _ => {
                    return Err(format!("Unexpected token at position {}: {:?}", self.pos, self.tokens[self.pos]));
                }
            }
        }
        
        Ok(())
    }
    
    fn emit_instruction(&mut self, name: &str) -> Result<(), String> {
        self.pos += 1;
        
        match name {
            "nop" => {
                self.program.bits.push(5, Opcode::Nop as u32);
            }
            
            "kill" => {
                self.program.push_kill();
            }
            
            "mov" => {
                let dest = self.expect_register()?;
                self.expect_comma()?;
                let value = self.expect_number()?;
                self.program.push_load_float(dest, value);
            }
            
            "add" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_add(ra, rb);
            }
            
            "sub" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_sub(ra, rb);
            }
            
            "mul" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_mul(ra, rb);
            }
            
            "div" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_div(ra, rb);
            }
            
            "vec2add" => {
                let dest = self.expect_vector()?;
                self.expect_comma()?;
                let src = self.expect_vector()?;
                self.program.push_vec2_add(dest, src);
            }
            
            "vec2sub" => {
                let dest = self.expect_vector()?;
                self.expect_comma()?;
                let src = self.expect_vector()?;
                self.program.push_vec2_sub(dest, src);
            }
            
            "vec2mul" => {
                let vec = self.expect_vector()?;
                self.expect_comma()?;
                let scalar = self.expect_register()?;
                self.program.push_vec2_mul(vec, scalar);
            }
            
            "vec2norm" => {
                let vec = self.expect_vector()?;
                self.program.push_vec2_normalize(vec);
            }
            
            "vec2len" => {
                let vec = self.expect_vector()?;
                self.expect_comma()?;
                let dest = self.expect_register()?;
                self.program.push_vec2_length(vec, dest);
            }
            
            "sin" => {
                let reg = self.expect_register()?;
                self.program.push_sin(reg);
            }
            
            "cos" => {
                let reg = self.expect_register()?;
                self.program.push_cos(reg);
            }
            
            "sqrt" => {
                let reg = self.expect_register()?;
                self.program.push_sqrt(reg);
            }
            
            "cmplt" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_cmp_lt(ra, rb);
            }
            
            "cmpgt" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_cmp_gt(ra, rb);
            }
            
            "cmpeq" => {
                let ra = self.expect_register()?;
                self.expect_comma()?;
                let rb = self.expect_register()?;
                self.program.push_cmp_eq(ra, rb);
            }
            
            "jmp" => {
                if let Token::LabelRef(label) = &self.tokens[self.pos] {
                    let fixup_pos = self.program.bits.len() + 5;
                    self.fixups.push((fixup_pos, label.clone()));
                    self.program.push_jump(0);
                    self.pos += 1;
                } else {
                    return Err("Expected label reference after jmp".to_string());
                }
            }
            
            "jz" => {
                let reg = self.expect_register()?;
                self.expect_comma()?;
                if let Token::LabelRef(label) = &self.tokens[self.pos] {
                    let fixup_pos = self.program.bits.len() + 5 + 3;
                    self.fixups.push((fixup_pos, label.clone()));
                    self.program.push_jump_if_zero(reg, 0);
                    self.pos += 1;
                } else {
                    return Err("Expected label reference".to_string());
                }
            }
            
            "jnz" => {
                let reg = self.expect_register()?;
                self.expect_comma()?;
                if let Token::LabelRef(label) = &self.tokens[self.pos] {
                    let fixup_pos = self.program.bits.len() + 5 + 3;
                    self.fixups.push((fixup_pos, label.clone()));
                    self.program.push_jump_if_not_zero(reg, 0);
                    self.pos += 1;
                } else {
                    return Err("Expected label reference".to_string());
                }
            }
            
            "jneg" => {
                let reg = self.expect_register()?;
                self.expect_comma()?;
                if let Token::LabelRef(label) = &self.tokens[self.pos] {
                    let fixup_pos = self.program.bits.len() + 5 + 3;
                    self.fixups.push((fixup_pos, label.clone()));
                    self.program.push_jump_if_neg(reg, 0);
                    self.pos += 1;
                } else {
                    return Err("Expected label reference".to_string());
                }
            }
            
            "jpos" => {
                let reg = self.expect_register()?;
                self.expect_comma()?;
                if let Token::LabelRef(label) = &self.tokens[self.pos] {
                    let fixup_pos = self.program.bits.len() + 5 + 3;
                    self.fixups.push((fixup_pos, label.clone()));
                    self.program.push_jump_if_pos(reg, 0);
                    self.pos += 1;
                } else {
                    return Err("Expected label reference".to_string());
                }
            }
            
            _ => {
                return Err(format!("Unknown instruction: {}", name));
            }
        }
        
        // Skip newline if present
        if self.pos < self.tokens.len() && matches!(self.tokens[self.pos], Token::Newline) {
            self.pos += 1;
        }
        
        Ok(())
    }
    
    fn expect_register(&mut self) -> Result<u8, String> {
        if let Token::Register(n) = self.tokens[self.pos] {
            self.pos += 1;
            Ok(n)
        } else {
            Err(format!("Expected register, got {:?}", self.tokens[self.pos]))
        }
    }
    
    fn expect_vector(&mut self) -> Result<u8, String> {
        if let Token::VectorPair(n) = self.tokens[self.pos] {
            self.pos += 1;
            Ok(n)
        } else {
            Err(format!("Expected vector, got {:?}", self.tokens[self.pos]))
        }
    }
    
    fn expect_number(&mut self) -> Result<f32, String> {
        if let Token::Number(n) = self.tokens[self.pos] {
            self.pos += 1;
            Ok(n)
        } else {
            Err(format!("Expected number, got {:?}", self.tokens[self.pos]))
        }
    }
    
    fn expect_comma(&mut self) -> Result<(), String> {
        if matches!(self.tokens[self.pos], Token::Comma) {
            self.pos += 1;
            Ok(())
        } else {
            Err("Expected comma".to_string())
        }
    }
    
    fn skip_instruction(&mut self) {
        self.pos += 1;
        while self.pos < self.tokens.len() {
            match &self.tokens[self.pos] {
                Token::Newline => {
                    self.pos += 1;
                    break;
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
    }
    
    fn estimate_instruction_size(&self, name: &str) -> Result<usize, String> {
        Ok(match name {
            "nop" | "kill" => 5,
            "mov" => 5 + 3 + 8,
            "add" | "sub" | "mul" | "div" => 5 + 3 + 3,
            "vec2add" | "vec2sub" => 5 + 2 + 2,
            "vec2mul" => 5 + 2 + 3,
            "vec2norm" => 5 + 2,
            "vec2len" => 5 + 2 + 3,
            "sin" | "cos" | "sqrt" => 5 + 3,
            "cmplt" | "cmpgt" | "cmpeq" => 5 + 3 + 3,
            "jmp" => 5 + 11,
            "jz" | "jnz" | "jneg" | "jpos" => 5 + 3 + 11,
            _ => return Err(format!("Unknown instruction: {}", name)),
        })
    }
    
    fn fixup_jumps(&mut self) -> Result<(), String> {
        for (bit_pos, label) in &self.fixups {
            let target = *self.labels.get(label)
                .ok_or(format!("Undefined label: {}", label))?;
            
            let offset = (target as i32) - (*bit_pos as i32 + 11);
            
            if offset < -1024 || offset > 1023 {
                return Err(format!("Jump offset too large: {} (max ±1023)", offset));
            }
            
            let offset_bits = (offset as u16) & 0x7FF;
            
            for i in 0..11 {
                let bit = (offset_bits >> i) & 1;
                let byte_idx = (*bit_pos + i) / 8;
                let bit_idx = (*bit_pos + i) % 8;
                
                if bit == 1 {
                    self.program.bits.data[byte_idx] |= 1 << bit_idx;
                } else {
                    self.program.bits.data[byte_idx] &= !(1 << bit_idx);
                }
            }
        }
        
        Ok(())
    }
}

// Helper function to assemble from source string
fn assemble(source: &str) -> Result<Program, String> {
    let mut lexer = Lexer::new(source.to_string());
    let tokens = lexer.tokenize()?;
    let mut assembler = Assembler::new(tokens);
    assembler.assemble()
}


// Simple bit container (replace with your packed_bits crate)
#[derive(Clone)] 
struct BitContainer {
    data: Vec<u8>,
    bit_len: usize,
}

impl BitContainer {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_len: 0,
        }
    }

    fn push(&mut self, num_bits: usize, value: u32) {
        for i in 0..num_bits {
            let bit = (value >> i) & 1;
            let byte_idx = self.bit_len / 8;
            let bit_idx = self.bit_len % 8;

            if byte_idx >= self.data.len() {
                self.data.push(0);
            }

            if bit == 1 {
                self.data[byte_idx] |= 1 << bit_idx;
            }

            self.bit_len += 1;
        }
    }

    fn get(&self, start_bit: usize, num_bits: usize) -> Option<u32> {
        if start_bit + num_bits > self.bit_len {
            return None;
        }

        let mut result = 0u32;
        for i in 0..num_bits {
            let byte_idx = (start_bit + i) / 8;
            let bit_idx = (start_bit + i) % 8;
            let bit = (self.data[byte_idx] >> bit_idx) & 1;
            result |= (bit as u32) << i;
        }

        Some(result)
    }

    fn len(&self) -> usize {
        self.bit_len
    }
}

#[derive(Clone)] 
struct Program {
    bits: BitContainer,
    float_constants: Vec<f32>,  // NEW: constant pool for floats
}

impl Program {
    fn new() -> Self {
        Self {
            bits: BitContainer::new(),
            float_constants: Vec::new(),
        }
    }

    fn push_mov_imm(&mut self, reg: u8, value: u8) {
        self.bits.push(5, Opcode::MovImm as u32);  // 5 bits for opcode now
        self.bits.push(3, reg as u32);              // 3 bits for 8 registers
        self.bits.push(8, value as u32);
    }
    
    fn push_load_float(&mut self, reg: u8, value: f32) {
        // Add to constant pool
        let const_idx = self.float_constants.len();
        self.float_constants.push(value);
        
        self.bits.push(5, Opcode::MovImm as u32);
        self.bits.push(3, reg as u32);
        self.bits.push(8, const_idx as u32);  // Index into constant pool
    }

    fn push_add(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::Add as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }

    fn push_sub(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::Sub as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }

    fn push_mul(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::Mul as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }
    
    fn push_div(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::Div as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }
    
    // Vector operations
    fn push_vec2_add(&mut self, dest_pair: u8, src_pair: u8) {
        // dest_pair: 0=R0-R1, 1=R2-R3, 2=R4-R5, 3=R6-R7
        self.bits.push(5, Opcode::Vec2Add as u32);
        self.bits.push(2, dest_pair as u32);
        self.bits.push(2, src_pair as u32);
    }
    
    fn push_vec2_sub(&mut self, dest_pair: u8, src_pair: u8) {
        self.bits.push(5, Opcode::Vec2Sub as u32);
        self.bits.push(2, dest_pair as u32);
        self.bits.push(2, src_pair as u32);
    }
    
    fn push_vec2_mul(&mut self, vec_pair: u8, scalar_reg: u8) {
        self.bits.push(5, Opcode::Vec2Mul as u32);
        self.bits.push(2, vec_pair as u32);
        self.bits.push(3, scalar_reg as u32);
    }
    
    fn push_vec2_normalize(&mut self, vec_pair: u8) {
        self.bits.push(5, Opcode::Vec2Normalize as u32);
        self.bits.push(2, vec_pair as u32);
    }
    
    fn push_vec2_length(&mut self, vec_pair: u8, dest_reg: u8) {
        self.bits.push(5, Opcode::Vec2Length as u32);
        self.bits.push(2, vec_pair as u32);
        self.bits.push(3, dest_reg as u32);
    }
    
    // Math functions
    fn push_sin(&mut self, reg: u8) {
        self.bits.push(5, Opcode::Sin as u32);
        self.bits.push(3, reg as u32);
    }
    
    fn push_cos(&mut self, reg: u8) {
        self.bits.push(5, Opcode::Cos as u32);
        self.bits.push(3, reg as u32);
    }
    
    fn push_sqrt(&mut self, reg: u8) {
        self.bits.push(5, Opcode::Sqrt as u32);
        self.bits.push(3, reg as u32);
    }

    fn push_jump(&mut self, offset: i16) {
        // offset is in bits, signed
        self.bits.push(5, Opcode::Jump as u32);
        self.bits.push(11, (offset as u16) as u32);  // 11 bits = ±1024 range
    }
    
    fn push_jump_if_zero(&mut self, reg: u8, offset: i16) {
        self.bits.push(5, Opcode::JumpIfZero as u32);
        self.bits.push(3, reg as u32);
        self.bits.push(11, (offset as u16) as u32);
    }
    
    fn push_jump_if_not_zero(&mut self, reg: u8, offset: i16) {
        self.bits.push(5, Opcode::JumpIfNotZero as u32);
        self.bits.push(3, reg as u32);
        self.bits.push(11, (offset as u16) as u32);
    }
    
    fn push_jump_if_neg(&mut self, reg: u8, offset: i16) {
        self.bits.push(5, Opcode::JumpIfNeg as u32);
        self.bits.push(3, reg as u32);
        self.bits.push(11, (offset as u16) as u32);
    }
    
    fn push_jump_if_pos(&mut self, reg: u8, offset: i16) {
        self.bits.push(5, Opcode::JumpIfPos as u32);
        self.bits.push(3, reg as u32);
        self.bits.push(11, (offset as u16) as u32);
    }
    
    fn push_cmp_lt(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::CmpLt as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }
    
    fn push_cmp_gt(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::CmpGt as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }
    
    fn push_cmp_eq(&mut self, ra: u8, rb: u8) {
        self.bits.push(5, Opcode::CmpEq as u32);
        self.bits.push(3, ra as u32);
        self.bits.push(3, rb as u32);
    }

    fn push_kill(&mut self) {
        self.bits.push(5, Opcode::Kill as u32);
    }
    
    // Helper to get current position (for calculating jump offsets)
    fn get_position(&self) -> usize {
        self.bits.len()
    }
}

#[derive(Debug)]
enum VMResult {
    Complete,
    Kill,
}

struct VM {
    regs: [f32; 8],  // Changed to f32 and expanded to 8 registers
}

impl VM {
    fn new() -> Self {
        Self { regs: [0.0; 8] }
    }

    fn run(&mut self, program: &Program) -> VMResult {
        let mut pc = 0;
        let max_iterations = 100000;  // Safety limit to prevent infinite loops
        let mut iteration_count = 0;

        while pc < program.bits.len() {
            iteration_count += 1;
            if iteration_count > max_iterations {
                println!("Warning: Hit iteration limit, possible infinite loop");
                break;
            }

            let opcode_bits = match program.bits.get(pc, 5) {  // 5 bits now
                Some(v) => v,
                None => break,
            };
            pc += 5;

            let opcode = match Opcode::from_u8(opcode_bits as u8) {
                Ok(op) => op,
                Err(e) => {
                    println!("Error: {}", e);
                    break;
                }
            };

            match opcode {
                Opcode::Nop => {}
                
                Opcode::Add => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[ra] += self.regs[rb];
                }
                
                Opcode::Sub => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[ra] -= self.regs[rb];
                }
                
                Opcode::Mul => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[ra] *= self.regs[rb];
                }
                
                Opcode::Div => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    if self.regs[rb] != 0.0 {
                        self.regs[ra] /= self.regs[rb];
                    }
                }
                
                Opcode::MovImm => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let const_idx = program.bits.get(pc, 8).unwrap() as usize;
                    pc += 8;
                    
                    // Load from constant pool if available, otherwise use as u8
                    if const_idx < program.float_constants.len() {
                        self.regs[reg] = program.float_constants[const_idx];
                    } else {
                        self.regs[reg] = const_idx as f32;
                    }
                }
                
                Opcode::MovReg => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[ra] = self.regs[rb];
                }
                
                Opcode::Vec2Add => {
                    let dest_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    let src_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    
                    self.regs[dest_pair] += self.regs[src_pair];
                    self.regs[dest_pair + 1] += self.regs[src_pair + 1];
                }
                
                Opcode::Vec2Sub => {
                    let dest_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    let src_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    
                    self.regs[dest_pair] -= self.regs[src_pair];
                    self.regs[dest_pair + 1] -= self.regs[src_pair + 1];
                }
                
                Opcode::Vec2Mul => {
                    let vec_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    let scalar_reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    
                    let scalar = self.regs[scalar_reg];
                    self.regs[vec_pair] *= scalar;
                    self.regs[vec_pair + 1] *= scalar;
                }
                
                Opcode::Vec2Length => {
                    let vec_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    let dest_reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    
                    let x = self.regs[vec_pair];
                    let y = self.regs[vec_pair + 1];
                    self.regs[dest_reg] = (x * x + y * y).sqrt();
                }
                
                Opcode::Vec2Normalize => {
                    let vec_pair = program.bits.get(pc, 2).unwrap() as usize * 2;
                    pc += 2;
                    
                    let x = self.regs[vec_pair];
                    let y = self.regs[vec_pair + 1];
                    let len = (x * x + y * y).sqrt();
                    
                    if len > 0.0001 {
                        self.regs[vec_pair] = x / len;
                        self.regs[vec_pair + 1] = y / len;
                    }
                }
                
                Opcode::Sin => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[reg] = self.regs[reg].sin();
                }
                
                Opcode::Cos => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[reg] = self.regs[reg].cos();
                }
                
                Opcode::Sqrt => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    self.regs[reg] = self.regs[reg].sqrt();
                }

                Opcode::Jump => {
                    let offset_bits = program.bits.get(pc, 11).unwrap();
                    // Sign extend 11-bit value
                    let offset = if offset_bits & 0x400 != 0 {
                        // Negative: sign extend
                        (offset_bits | 0xFFFFF800) as i32
                    } else {
                        offset_bits as i32
                    };
                    
                    // Jump is relative to the NEXT instruction
                    pc = (pc as i32 + 11 + offset) as usize;
                }
                
                Opcode::JumpIfZero => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let offset_bits = program.bits.get(pc, 11).unwrap();
                    pc += 11;
                    
                    if self.regs[reg].abs() < 0.0001 {  // Float comparison
                        let offset = if offset_bits & 0x400 != 0 {
                            (offset_bits | 0xFFFFF800) as i32
                        } else {
                            offset_bits as i32
                        };
                        pc = (pc as i32 + offset) as usize;
                    }
                }
                
                Opcode::JumpIfNotZero => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let offset_bits = program.bits.get(pc, 11).unwrap();
                    pc += 11;
                    
                    if self.regs[reg].abs() >= 0.0001 {
                        let offset = if offset_bits & 0x400 != 0 {
                            (offset_bits | 0xFFFFF800) as i32
                        } else {
                            offset_bits as i32
                        };
                        pc = (pc as i32 + offset) as usize;
                    }
                }
                
                Opcode::JumpIfNeg => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let offset_bits = program.bits.get(pc, 11).unwrap();
                    pc += 11;
                    
                    if self.regs[reg] < 0.0 {
                        let offset = if offset_bits & 0x400 != 0 {
                            (offset_bits | 0xFFFFF800) as i32
                        } else {
                            offset_bits as i32
                        };
                        pc = (pc as i32 + offset) as usize;
                    }
                }
                
                Opcode::JumpIfPos => {
                    let reg = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let offset_bits = program.bits.get(pc, 11).unwrap();
                    pc += 11;
                    
                    if self.regs[reg] > 0.0 {
                        let offset = if offset_bits & 0x400 != 0 {
                            (offset_bits | 0xFFFFF800) as i32
                        } else {
                            offset_bits as i32
                        };
                        pc = (pc as i32 + offset) as usize;
                    }
                }
                
                Opcode::CmpLt => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    
                    self.regs[ra] = if self.regs[ra] < self.regs[rb] { 1.0 } else { 0.0 };
                }
                
                Opcode::CmpGt => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    
                    self.regs[ra] = if self.regs[ra] > self.regs[rb] { 1.0 } else { 0.0 };
                }
                
                Opcode::CmpEq => {
                    let ra = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    let rb = program.bits.get(pc, 3).unwrap() as usize;
                    pc += 3;
                    
                    self.regs[ra] = if (self.regs[ra] - self.regs[rb]).abs() < 0.0001 { 1.0 } else { 0.0 };
                }
                
                Opcode::Kill => {
                    return VMResult::Kill;
                } 
            }
        }

        VMResult::Complete
    }
}

fn main() {
    println!("=== Enhanced VM with Floats & Vectors ===\n");

    // Test 1: Basic arithmetic
    println!("Test 1: Basic arithmetic");
    let mut program = Program::new();
    program.push_load_float(0, 10.5);
    program.push_load_float(1, 20.3);
    program.push_add(0, 1);
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  10.5 + 20.3 = {}", vm.regs[0]);
    println!("  Expected: ~30.8\n");

    // Test 2: Vector operations
    println!("Test 2: Vector addition");
    let mut program = Program::new();
    // Set up vector 1 (R0-R1)
    program.push_load_float(0, 3.0);  // x
    program.push_load_float(1, 4.0);  // y
    // Set up vector 2 (R2-R3)
    program.push_load_float(2, 1.0);  // x
    program.push_load_float(3, 2.0);  // y
    // Add them
    program.push_vec2_add(0, 1);  // v0 += v1
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  (3, 4) + (1, 2) = ({}, {})", vm.regs[0], vm.regs[1]);
    println!("  Expected: (4, 6)\n");

    // Test 3: Vector normalize
    println!("Test 3: Vector normalize");
    let mut program = Program::new();
    program.push_load_float(0, 3.0);
    program.push_load_float(1, 4.0);
    program.push_vec2_normalize(0);
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  normalize(3, 4) = ({:.3}, {:.3})", vm.regs[0], vm.regs[1]);
    println!("  Expected: (0.6, 0.8)\n");

    // Test 4: Vector length and scaling
    println!("Test 4: Vector length and scaling");
    let mut program = Program::new();
    program.push_load_float(0, 3.0);
    program.push_load_float(1, 4.0);
    program.push_vec2_length(0, 4);      // Store length in R4
    program.push_load_float(5, 2.0);     // Scale factor
    program.push_vec2_mul(0, 5);         // Multiply vector by 2
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  length(3, 4) = {}", vm.regs[4]);
    println!("  (3, 4) * 2 = ({}, {})", vm.regs[0], vm.regs[1]);
    println!("  Expected: length=5, scaled=(6, 8)\n");

    // Test 5: Circular motion (like pinwheel enemy)
    println!("Test 5: Circular motion");
    let mut program = Program::new();
    let angle = std::f32::consts::PI / 4.0; // 45 degrees
    program.push_load_float(4, angle);
    program.push_cos(4);                 // R4 = cos(angle)
    program.push_load_float(5, angle);
    program.push_sin(5);                 // R5 = sin(angle)
    program.push_load_float(6, 10.0);    // Speed
    program.push_mul(4, 6);              // R4 *= speed
    program.push_mul(5, 6);              // R5 *= speed
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  velocity at 45° with speed 10:");
    println!("  x = {:.3}, y = {:.3}", vm.regs[4], vm.regs[5]);
    println!("  Expected: (~7.07, ~7.07)\n");

    // Test 6: Particle behavior simulation
    println!("Test 6: Simple particle update (position += velocity * dt)");
    let mut program = Program::new();
    // Initial position (R0-R1)
    program.push_load_float(0, 100.0);
    program.push_load_float(1, 100.0);
    // Velocity (R2-R3)
    program.push_load_float(2, 10.0);
    program.push_load_float(3, 5.0);
    // Delta time
    program.push_load_float(4, 0.016);   // 60fps
    // velocity *= dt
    program.push_vec2_mul(1, 4);         // v1 (R2-R3) *= R4
    // position += velocity
    program.push_vec2_add(0, 1);         // v0 (R0-R1) += v1 (R2-R3)
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Start: pos(100, 100), vel(10, 5), dt=0.016");
    println!("  After update: pos({:.3}, {:.3})", vm.regs[0], vm.regs[1]);
    println!("  Expected: (~100.16, ~100.08)\n");

    println!("Program sizes:");
    println!("  Test 6: {} bits ({} bytes)", 
             program.bits.len(),
             (program.bits.len() + 7) / 8);
    
    println!("\n✓ All tests complete!");


    println!("=== VM with Control Flow ===\n");

    // Test 1: Simple conditional (if distance < 10, kill)
    println!("Test 1: Conditional kill");
    let mut program = Program::new();
    program.push_load_float(0, 5.0);   // distance
    program.push_load_float(1, 10.0);  // threshold
    program.push_cmp_lt(0, 1);         // R0 = (distance < 10) ? 1 : 0
    program.push_jump_if_zero(0, 16);  // Skip kill if false (16 bits = kill instruction)
    program.push_kill();
    
    let mut vm = VM::new();
    let result = vm.run(&program);
    println!("  Distance 5 < 10: {:?}", result);
    println!("  Expected: Kill\n");

    // Test 2: Loop (countdown from 5 to 0)
    println!("Test 2: Loop countdown");
    let mut program = Program::new();
    program.push_load_float(0, 5.0);   // Counter
    program.push_load_float(1, 1.0);   // Decrement
    // Loop start (position = 32 bits)
    let loop_start = program.get_position();
    program.push_sub(0, 1);            // counter -= 1
    program.push_jump_if_pos(0, -19);  // Jump back if counter > 0
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Final counter: {}", vm.regs[0]);
    println!("  Expected: 0\n");

    // Test 3: Distance check (enemy AI behavior)
    println!("Test 3: Enemy distance check");
    let mut program = Program::new();
    // Player position (R0-R1)
    program.push_load_float(0, 100.0);
    program.push_load_float(1, 100.0);
    // Enemy position (R2-R3)
    program.push_load_float(2, 105.0);
    program.push_load_float(3, 103.0);
    // Calculate direction: player - enemy
    program.push_vec2_sub(0, 1);       // R0-R1 = player - enemy
    // Get distance
    program.push_vec2_length(0, 4);    // R4 = length
    // Check if distance < 10
    program.push_load_float(5, 10.0);
    program.push_cmp_lt(4, 5);         // R4 = (distance < 10) ? 1 : 0
    program.push_jump_if_zero(4, 16);  // Skip if false
    program.push_load_float(6, 1.0);   // Set "should_chase" flag
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Distance: {:.2}", vm.regs[4]);
    println!("  Should chase: {}", vm.regs[6] != 0.0);
    println!("  Expected: distance ~6.4, should_chase true\n");

    // Test 4: Particle lifetime check
    println!("Test 4: Particle lifetime");
    let mut program = Program::new();
    program.push_load_float(0, 0.5);   // Lifetime remaining
    program.push_load_float(1, 0.016); // dt (one frame at 60fps)
    program.push_sub(0, 1);            // lifetime -= dt
    program.push_jump_if_neg(0, 8);    // Jump to kill if negative
    // Particle still alive, do update
    program.push_load_float(2, 1.0);   // alive = 1
    program.push_jump(8);              // Jump over kill
    // Kill particle
    program.push_kill();
    
    let mut vm = VM::new();
    let result = vm.run(&program);
    println!("  After 1 frame: lifetime={:.3}, status={:?}", vm.regs[0], result);
    
    // Run again with more frames
    let mut program = Program::new();
    program.push_load_float(0, 0.02);  // Small lifetime
    program.push_load_float(1, 0.016);
    program.push_sub(0, 1);
    program.push_jump_if_neg(0, 8);
    program.push_load_float(2, 1.0);
    program.push_jump(8);
    program.push_kill();
    
    let mut vm = VM::new();
    let result = vm.run(&program);
    println!("  After 1 frame (low lifetime): lifetime={:.3}, status={:?}", vm.regs[0], result);
    println!("  Expected: First alive, second Kill\n");

    // Test 5: State machine (patrol vs chase)
    println!("Test 5: AI state machine");
    let mut program = Program::new();
    program.push_load_float(0, 8.0);   // distance to player
    program.push_load_float(1, 10.0);  // chase threshold
    program.push_cmp_lt(0, 1);         // distance < threshold?
    program.push_jump_if_zero(0, 24);  // If false, go to patrol state
    // Chase state
    program.push_load_float(2, 2.0);   // chase_speed
    program.push_jump(16);             // Skip patrol
    // Patrol state
    program.push_load_float(2, 1.0);   // patrol_speed
    // End
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Distance 8 < 10: speed = {}", vm.regs[2]);
    println!("  Expected: 2.0 (chase speed)\n");

    println!("✓ All control flow tests complete!");
    println!("\nYou can now:");
    println!("  - Write if/else statements");
    println!("  - Create loops");
    println!("  - Build state machines");
    println!("  - Implement complex AI behaviors");


    println!("=== VM Assembler Test ===\n");

    // Test 1: Simple arithmetic
    println!("Test 1: Simple arithmetic");
    let source = r#"
        mov r0, 10.5
        mov r1, 20.3
        add r0, r1
    "#;
    
    let program = assemble(source).expect("Assembly failed");
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Result: {:.1}", vm.regs[0]);
    println!("  Expected: 30.8\n");

    // Test 2: Vector operations
    println!("Test 2: Vector operations");
    let source = r#"
        ; Setup vector (3, 4)
        mov r0, 3.0
        mov r1, 4.0
        ; Normalize it
        vec2norm v0
    "#;
    
    let program = assemble(source).expect("Assembly failed");
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Normalized (3, 4) = ({:.2}, {:.2})", vm.regs[0], vm.regs[1]);
    println!("  Expected: (0.60, 0.80)\n");

    // Test 3: Loop with label
    println!("Test 3: Loop (countdown)");
    let source = r#"
        mov r0, 5.0
        mov r1, 1.0
    loop:
        sub r0, r1
        jpos r0, @loop
    "#;
    
    let program = assemble(source).expect("Assembly failed");
    let mut vm = VM::new();
    vm.run(&program);
    println!("  Final counter: {}", vm.regs[0]);
    println!("  Expected: 0\n");

    // Test 4: Conditional
    println!("Test 4: Distance check");
    let source = r#"
        ; Check if distance < 10
        mov r0, 5.0
        mov r1, 10.0
        cmplt r0, r1
        jz r0, @skip
        mov r2, 1.0   ; Set flag
    skip:
    "#;
    
    let program = assemble(source).expect("Assembly failed");
    let mut vm = VM::new();
    vm.run(&program);
    println!("  5 < 10, flag = {}", vm.regs[2]);
    println!("  Expected: 1\n");

    // Test 5: Particle behavior
    println!("Test 5: Particle update");
    let source = r#"
        ; Position (R0-R1), Velocity (R2-R3)
        mov r0, 100.0
        mov r1, 100.0
        mov r2, 10.0
        mov r3, 5.0
        
        ; dt
        mov r4, 0.016
        
        ; velocity *= dt
        vec2mul v1, r4
        
        ; position += velocity
        vec2add v0, v1
    "#;
    
    let program = assemble(source).expect("Assembly failed");
    println!("  Program size: {} bits ({} bytes)", 
             program.bits.len(),
             (program.bits.len() + 7) / 8);
    
    let mut vm = VM::new();
    vm.run(&program);
    println!("  New position: ({:.2}, {:.2})", vm.regs[0], vm.regs[1]);
    println!("  Expected: (~100.16, ~100.08)\n");

    println!("✓ All assembler tests passed!");
    println!("\nYou can now write assembly programs!");
}
// #[derive(Debug, Clone, Copy)]
// enum Opcode {
//     // === Control Flow (3 bits) ===
//     Nop = 0,
//     Kill = 1,           // Destroy this entity
//     Jump = 2,           // Unconditional jump
//     JumpIfZero = 3,     // Jump if reg == 0
//     JumpIfNeg = 4,      // Jump if reg < 0
    
//     // === Data Movement (3 bits) ===
//     MovImm = 5,         // reg = immediate (8-bit)
//     MovReg = 6,         // reg = reg
//     LoadFloat = 7,      // Load 32-bit float from const table
    
//     // === Scalar Arithmetic (3 bits) ===
//     Add = 8,            // ra += rb
//     Sub = 9,            // ra -= rb
//     Mul = 10,           // ra *= rb
//     Div = 11,           // ra /= rb
//     MulImm = 12,        // reg *= immediate (common for dt scaling)
    
//     // === Vector Operations (3 bits) ===
//     // Work on register pairs (R0-R1, R2-R3, R4-R5, R6-R7)
//     Vec2Add = 16,       // vec_a += vec_b
//     Vec2Sub = 17,       // vec_a -= vec_b
//     Vec2Mul = 18,       // vec_a *= scalar
//     Vec2Length = 19,    // scalar = length(vec)
//     Vec2Normalize = 20, // vec = normalize(vec)
//     Vec2Dot = 21,       // scalar = dot(vec_a, vec_b)
    
//     // === Math Functions (3 bits) ===
//     Sin = 24,           // reg = sin(reg)
//     Cos = 25,           // reg = cos(reg)
//     Sqrt = 26,          // reg = sqrt(reg)
//     Lerp = 27,          // ra = lerp(ra, rb, rc) where rc is t
//     Abs = 28,           // reg = abs(reg)
    
//     // === Comparison (3 bits) ===
//     CmpLt = 32,         // reg = (ra < rb) ? 1 : 0
//     CmpGt = 33,         // reg = (ra > rb) ? 1 : 0
//     CmpEq = 34,         // reg = (ra == rb) ? 1 : 0
    
//     // === Game-Specific (3 bits) ===
//     GetDeltaTime = 40,  // reg = dt
//     GetPlayerPos = 41,  // vec = player position
//     GetSelfPos = 42,    // vec = self position
//     GetRandom = 43,     // reg = random [0,1)
//     SpawnParticle = 44, // Spawn particle at position
//     GetLifetime = 45,   // reg = remaining lifetime
// }

// // Example bit layouts:

// // Scalar ops: [opcode:5][dest:3][src:3] = 11 bits
// // Vector ops: [opcode:5][vec_pair_dest:2][vec_pair_src:2] = 9 bits
// // Immediate: [opcode:5][dest:3][immediate:8] = 16 bits
// // Jump: [opcode:5][offset:11] = 16 bits (±1024 instruction offset)
// // No-arg: [opcode:5] = 5 bits

// struct Program {
//     bits: BitContainer,
//     float_constants: Vec<f32>, // For precise values like PI, speeds, etc.
// }

// impl Program {
//     fn push_vec2_add(&mut self, dest_pair: u8, src_pair: u8) {
//         // dest_pair: 0=R0-R1, 1=R2-R3, 2=R4-R5, 3=R6-R7
//         self.bits.push(5, Opcode::Vec2Add as u32).unwrap();
//         self.bits.push(2, dest_pair as u32).unwrap();
//         self.bits.push(2, src_pair as u32).unwrap();
//     }
    
//     fn push_mul_imm(&mut self, reg: u8, value: u8) {
//         // For small multipliers (like dt * 60 ≈ 1.0)
//         self.bits.push(5, Opcode::MulImm as u32).unwrap();
//         self.bits.push(3, reg as u32).unwrap();
//         self.bits.push(8, value as u32).unwrap();
//     }
    
//     fn push_load_float(&mut self, reg: u8, const_idx: u8) {
//         // Load from constant table
//         self.bits.push(5, Opcode::LoadFloat as u32).unwrap();
//         self.bits.push(3, reg as u32).unwrap();
//         self.bits.push(8, const_idx as u32).unwrap();
//     }
// }

// struct VM {
//     regs: [f32; 8],  // Changed to f32 for game math
//     pc: usize,       // Program counter (bit index)
    
//     // Context passed from game engine
//     dt: f32,
//     player_pos: Vec2,
//     self_pos: Vec2,
//     lifetime: f32,
//     rng: Random,
// }

// impl VM {
//     fn run(&mut self, program: &Program) -> VMResult {
//         loop {
//             if self.pc >= program.bits.len() {
//                 return VMResult::Complete;
//             }
            
//             let opcode = program.bits.get(self.pc, 5).unwrap() as u8;
//             self.pc += 5;
            
//             match Opcode::from_u8(opcode) {
//                 Opcode::Kill => return VMResult::Kill,
                
//                 Opcode::Vec2Add => {
//                     let dest = program.bits.get(self.pc, 2).unwrap() as usize * 2;
//                     self.pc += 2;
//                     let src = program.bits.get(self.pc, 2).unwrap() as usize * 2;
//                     self.pc += 2;
                    
//                     self.regs[dest] += self.regs[src];
//                     self.regs[dest + 1] += self.regs[src + 1];
//                 }
                
//                 Opcode::Vec2Normalize => {
//                     let pair = program.bits.get(self.pc, 2).unwrap() as usize * 2;
//                     self.pc += 2;
                    
//                     let x = self.regs[pair];
//                     let y = self.regs[pair + 1];
//                     let len = (x * x + y * y).sqrt();
                    
//                     if len > 0.0001 {
//                         self.regs[pair] = x / len;
//                         self.regs[pair + 1] = y / len;
//                     }
//                 }
                
//                 Opcode::GetDeltaTime => {
//                     let reg = program.bits.get(self.pc, 3).unwrap() as usize;
//                     self.pc += 3;
//                     self.regs[reg] = self.dt;
//                 }
                
//                 Opcode::MulImm => {
//                     let reg = program.bits.get(self.pc, 3).unwrap() as usize;
//                     self.pc += 3;
//                     let imm = program.bits.get(self.pc, 8).unwrap() as f32 / 255.0;
//                     self.pc += 8;
//                     self.regs[reg] *= imm;
//                 }
                
//                 // ... other opcodes
//             }
//         }
//     }
// }

// enum VMResult {
//     Complete,
//     Kill,
// }


// #[derive(Debug, Clone, PartialEq)]
// enum Token {
//     // Instructions
//     Instruction(String),
    
//     // Registers
//     Register(u8),           // r0-r7
//     VectorPair(u8),        // v0-v3 (maps to r0-r1, r2-r3, r4-r5, r6-r7)
    
//     // Literals
//     Integer(i32),
//     Float(f32),
    
//     // Labels
//     Label(String),         // "loop:"
//     LabelRef(String),      // "@loop"
    
//     // Directives
//     Const(String, f32),    // .const NAME value
    
//     // Misc
//     Comma,
//     Newline,
//     Comment,
// }

// struct Lexer {
//     input: String,
//     pos: usize,
// }

// impl Lexer {
//     fn new(input: String) -> Self {
//         Self { input, pos: 0 }
//     }
    
//     fn tokenize(&mut self) -> Result<Vec<Token>, String> {
//         let mut tokens = Vec::new();
        
//         while self.pos < self.input.len() {
//             self.skip_whitespace();
            
//             if self.pos >= self.input.len() {
//                 break;
//             }
            
//             let ch = self.current_char();
            
//             match ch {
//                 ';' => {
//                     self.skip_line();
//                     continue;
//                 }
//                 ',' => {
//                     tokens.push(Token::Comma);
//                     self.advance();
//                 }
//                 '.' => {
//                     // Directive like .const
//                     self.advance();
//                     let directive = self.read_identifier();
//                     if directive == "const" {
//                         self.skip_whitespace();
//                         let name = self.read_identifier();
//                         self.skip_whitespace();
//                         let value = self.read_number()?;
//                         tokens.push(Token::Const(name, value));
//                     }
//                 }
//                 '@' => {
//                     // Label reference
//                     self.advance();
//                     let name = self.read_identifier();
//                     tokens.push(Token::LabelRef(name));
//                 }
//                 _ if ch.is_alphabetic() => {
//                     let ident = self.read_identifier();
                    
//                     // Check if it's a label definition
//                     if self.peek_char() == Some(':') {
//                         self.advance(); // consume ':'
//                         tokens.push(Token::Label(ident));
//                     }
//                     // Check for register
//                     else if ident.starts_with('r') {
//                         let num = ident[1..].parse::<u8>()
//                             .map_err(|_| format!("Invalid register: {}", ident))?;
//                         tokens.push(Token::Register(num));
//                     }
//                     // Check for vector pair
//                     else if ident.starts_with('v') {
//                         let num = ident[1..].parse::<u8>()
//                             .map_err(|_| format!("Invalid vector: {}", ident))?;
//                         tokens.push(Token::VectorPair(num));
//                     }
//                     // Otherwise it's an instruction
//                     else {
//                         tokens.push(Token::Instruction(ident));
//                     }
//                 }
//                 _ if ch.is_numeric() || ch == '-' => {
//                     let num = self.read_number()?;
//                     if num.fract() == 0.0 {
//                         tokens.push(Token::Integer(num as i32));
//                     } else {
//                         tokens.push(Token::Float(num));
//                     }
//                 }
//                 '\n' => {
//                     tokens.push(Token::Newline);
//                     self.advance();
//                 }
//                 _ => {
//                     return Err(format!("Unexpected character: {}", ch));
//                 }
//             }
//         }
        
//         Ok(tokens)
//     }
    
//     fn current_char(&self) -> char {
//         self.input.chars().nth(self.pos).unwrap()
//     }
    
//     fn peek_char(&self) -> Option<char> {
//         self.input.chars().nth(self.pos + 1)
//     }
    
//     fn advance(&mut self) {
//         self.pos += 1;
//     }
    
//     fn skip_whitespace(&mut self) {
//         while self.pos < self.input.len() {
//             let ch = self.current_char();
//             if ch == ' ' || ch == '\t' || ch == '\r' {
//                 self.advance();
//             } else {
//                 break;
//             }
//         }
//     }
    
//     fn skip_line(&mut self) {
//         while self.pos < self.input.len() && self.current_char() != '\n' {
//             self.advance();
//         }
//     }
    
//     fn read_identifier(&mut self) -> String {
//         let start = self.pos;
//         while self.pos < self.input.len() {
//             let ch = self.current_char();
//             if ch.is_alphanumeric() || ch == '_' {
//                 self.advance();
//             } else {
//                 break;
//             }
//         }
//         self.input[start..self.pos].to_string()
//     }
    
//     fn read_number(&mut self) -> Result<f32, String> {
//         let start = self.pos;
        
//         if self.current_char() == '-' {
//             self.advance();
//         }
        
//         while self.pos < self.input.len() && 
//               (self.current_char().is_numeric() || self.current_char() == '.') {
//             self.advance();
//         }
        
//         self.input[start..self.pos].parse::<f32>()
//             .map_err(|e| format!("Invalid number: {}", e))
//     }
// }

// struct Assembler {
//     tokens: Vec<Token>,
//     pos: usize,
//     program: Program,
//     labels: HashMap<String, usize>,  // label name -> bit position
//     fixups: Vec<(usize, String)>,    // (bit_pos, label_name) for jumps
//     constants: HashMap<String, usize>, // const name -> index in float table
// }

// impl Assembler {
//     fn new(tokens: Vec<Token>) -> Self {
//         Self {
//             tokens,
//             pos: 0,
//             program: Program::new(),
//             labels: HashMap::new(),
//             fixups: Vec::new(),
//             constants: HashMap::new(),
//         }
//     }
    
//     fn assemble(&mut self) -> Result<Program, String> {
//         // First pass: collect labels and constants
//         self.first_pass()?;
        
//         // Second pass: emit code
//         self.pos = 0;
//         self.second_pass()?;
        
//         // Third pass: fix up jump targets
//         self.fixup_jumps()?;
        
//         Ok(self.program.clone())
//     }
    
//     fn first_pass(&mut self) -> Result<(), String> {
//         let mut bit_pos = 0;
        
//         while self.pos < self.tokens.len() {
//             match &self.tokens[self.pos] {
//                 Token::Label(name) => {
//                     self.labels.insert(name.clone(), bit_pos);
//                 }
//                 Token::Const(name, value) => {
//                     let idx = self.program.float_constants.len();
//                     self.program.float_constants.push(*value);
//                     self.constants.insert(name.clone(), idx);
//                 }
//                 Token::Instruction(name) => {
//                     // Calculate instruction size
//                     bit_pos += self.estimate_instruction_size(name)?;
//                 }
//                 Token::Newline | Token::Comment => {}
//                 _ => {}
//             }
//             self.pos += 1;
//         }
        
//         Ok(())
//     }
    
//     fn second_pass(&mut self) -> Result<(), String> {
//         while self.pos < self.tokens.len() {
//             match &self.tokens[self.pos].clone() {
//                 Token::Instruction(name) => {
//                     self.emit_instruction(name)?;
//                 }
//                 Token::Newline | Token::Comment | Token::Label(_) | Token::Const(_, _) => {
//                     self.pos += 1;
//                 }
//                 _ => {
//                     return Err(format!("Unexpected token: {:?}", self.tokens[self.pos]));
//                 }
//             }
//         }
        
//         Ok(())
//     }
    
//     fn emit_instruction(&mut self, name: &str) -> Result<(), String> {
//         self.pos += 1; // Move past instruction
        
//         match name {
//             "nop" => {
//                 self.program.bits.push(5, Opcode::Nop as u32).unwrap();
//             }
            
//             "kill" => {
//                 self.program.bits.push(5, Opcode::Kill as u32).unwrap();
//             }
            
//             "add" => {
//                 // add r0, r1
//                 let dest = self.expect_register()?;
//                 self.expect_comma()?;
//                 let src = self.expect_register()?;
                
//                 self.program.bits.push(5, Opcode::Add as u32).unwrap();
//                 self.program.bits.push(3, dest as u32).unwrap();
//                 self.program.bits.push(3, src as u32).unwrap();
//             }
            
//             "mul" => {
//                 let dest = self.expect_register()?;
//                 self.expect_comma()?;
//                 let src = self.expect_register()?;
                
//                 self.program.bits.push(5, Opcode::Mul as u32).unwrap();
//                 self.program.bits.push(3, dest as u32).unwrap();
//                 self.program.bits.push(3, src as u32).unwrap();
//             }
            
//             "vec2add" => {
//                 // vec2add v0, v1
//                 let dest = self.expect_vector()?;
//                 self.expect_comma()?;
//                 let src = self.expect_vector()?;
                
//                 self.program.bits.push(5, Opcode::Vec2Add as u32).unwrap();
//                 self.program.bits.push(2, dest as u32).unwrap();
//                 self.program.bits.push(2, src as u32).unwrap();
//             }
            
//             "vec2sub" => {
//                 let dest = self.expect_vector()?;
//                 self.expect_comma()?;
//                 let src = self.expect_vector()?;
                
//                 self.program.bits.push(5, Opcode::Vec2Sub as u32).unwrap();
//                 self.program.bits.push(2, dest as u32).unwrap();
//                 self.program.bits.push(2, src as u32).unwrap();
//             }
            
//             "vec2mul" => {
//                 // vec2mul v0, r4 (multiply vector by scalar)
//                 let vec = self.expect_vector()?;
//                 self.expect_comma()?;
//                 let scalar = self.expect_register()?;
                
//                 self.program.bits.push(5, Opcode::Vec2Mul as u32).unwrap();
//                 self.program.bits.push(2, vec as u32).unwrap();
//                 self.program.bits.push(3, scalar as u32).unwrap();
//             }
            
//             "vec2norm" => {
//                 let vec = self.expect_vector()?;
                
//                 self.program.bits.push(5, Opcode::Vec2Normalize as u32).unwrap();
//                 self.program.bits.push(2, vec as u32).unwrap();
//             }
            
//             "loadf" => {
//                 // loadf r0, SPEED (load float constant)
//                 let reg = self.expect_register()?;
//                 self.expect_comma()?;
                
//                 let const_idx = if let Token::Instruction(name) = &self.tokens[self.pos] {
//                     // It's a named constant
//                     let idx = *self.constants.get(name)
//                         .ok_or(format!("Unknown constant: {}", name))?;
//                     self.pos += 1;
//                     idx
//                 } else {
//                     return Err("Expected constant name".to_string());
//                 };
                
//                 self.program.bits.push(5, Opcode::LoadFloat as u32).unwrap();
//                 self.program.bits.push(3, reg as u32).unwrap();
//                 self.program.bits.push(8, const_idx as u32).unwrap();
//             }
            
//             "getdt" => {
//                 let reg = self.expect_register()?;
                
//                 self.program.bits.push(5, Opcode::GetDeltaTime as u32).unwrap();
//                 self.program.bits.push(3, reg as u32).unwrap();
//             }
            
//             "getplayerpos" => {
//                 let vec = self.expect_vector()?;
                
//                 self.program.bits.push(5, Opcode::GetPlayerPos as u32).unwrap();
//                 self.program.bits.push(2, vec as u32).unwrap();
//             }
            
//             "getselfpos" => {
//                 let vec = self.expect_vector()?;
                
//                 self.program.bits.push(5, Opcode::GetSelfPos as u32).unwrap();
//                 self.program.bits.push(2, vec as u32).unwrap();
//             }
            
//             "jump" => {
//                 // jump @label
//                 if let Token::LabelRef(label) = &self.tokens[self.pos] {
//                     let fixup_pos = self.program.bits.len();
//                     self.fixups.push((fixup_pos + 5, label.clone())); // +5 for opcode
                    
//                     self.program.bits.push(5, Opcode::Jump as u32).unwrap();
//                     self.program.bits.push(11, 0).unwrap(); // Placeholder
                    
//                     self.pos += 1;
//                 } else {
//                     return Err("Expected label reference".to_string());
//                 }
//             }
            
//             "sin" => {
//                 let reg = self.expect_register()?;
//                 self.program.bits.push(5, Opcode::Sin as u32).unwrap();
//                 self.program.bits.push(3, reg as u32).unwrap();
//             }
            
//             "cos" => {
//                 let reg = self.expect_register()?;
//                 self.program.bits.push(5, Opcode::Cos as u32).unwrap();
//                 self.program.bits.push(3, reg as u32).unwrap();
//             }
            
//             _ => {
//                 return Err(format!("Unknown instruction: {}", name));
//             }
//         }
        
//         Ok(())
//     }
    
//     fn expect_register(&mut self) -> Result<u8, String> {
//         if let Token::Register(n) = self.tokens[self.pos] {
//             self.pos += 1;
//             Ok(n)
//         } else {
//             Err(format!("Expected register, got {:?}", self.tokens[self.pos]))
//         }
//     }
    
//     fn expect_vector(&mut self) -> Result<u8, String> {
//         if let Token::VectorPair(n) = self.tokens[self.pos] {
//             self.pos += 1;
//             Ok(n)
//         } else {
//             Err(format!("Expected vector, got {:?}", self.tokens[self.pos]))
//         }
//     }
    
//     fn expect_comma(&mut self) -> Result<(), String> {
//         if matches!(self.tokens[self.pos], Token::Comma) {
//             self.pos += 1;
//             Ok(())
//         } else {
//             Err("Expected comma".to_string())
//         }
//     }
    
//     fn estimate_instruction_size(&self, name: &str) -> Result<usize, String> {
//         // Return size in bits
//         Ok(match name {
//             "nop" | "kill" => 5,
//             "add" | "sub" | "mul" | "div" => 5 + 3 + 3,  // opcode + 2 regs
//             "vec2add" | "vec2sub" => 5 + 2 + 2,          // opcode + 2 vecs
//             "vec2mul" => 5 + 2 + 3,                       // opcode + vec + reg
//             "vec2norm" => 5 + 2,                          // opcode + vec
//             "loadf" => 5 + 3 + 8,                         // opcode + reg + const_idx
//             "getdt" | "sin" | "cos" => 5 + 3,            // opcode + reg
//             "getplayerpos" | "getselfpos" => 5 + 2,      // opcode + vec
//             "jump" => 5 + 11,                             // opcode + offset
//             _ => return Err(format!("Unknown instruction: {}", name)),
//         })
//     }
    
//     fn fixup_jumps(&mut self) -> Result<(), String> {
//         for (bit_pos, label) in &self.fixups {
//             let target = *self.labels.get(label)
//                 .ok_or(format!("Undefined label: {}", label))?;
            
//             let offset = (target as i32) - (*bit_pos as i32);
            
//             if offset < -1024 || offset > 1023 {
//                 return Err(format!("Jump offset too large: {}", offset));
//             }
            
//             // Patch the offset into the bitstream
//             let offset_bits = (offset as u32) & 0x7FF;
//             self.program.bits.set(*bit_pos, 11, offset_bits)?;
//         }
        
//         Ok(())
//     }
// }

// // Helper for BitContainer::set (you'll need to add this)
// impl BitContainer {
//     fn set(&mut self, start_bit: usize, num_bits: usize, value: u32) -> Result<(), String> {
//         // Overwrite bits at position
//         for i in 0..num_bits {
//             let bit = (value >> i) & 1;
//             let byte_idx = (start_bit + i) / 8;
//             let bit_idx = (start_bit + i) % 8;
            
//             if bit == 1 {
//                 self.data[byte_idx] |= 1 << bit_idx;
//             } else {
//                 self.data[byte_idx] &= !(1 << bit_idx);
//             }
//         }
//         Ok(())
//     }
// }

// fn main() {
//     let source = r#"
//         .const SPEED 150.0
//         .const DAMPING 0.95
        
//         ; Simple particle with damping
//         loadf r4, DAMPING
//         vec2mul v1, r4          ; velocity *= damping
        
//         getdt r4
//         vec2mul v1, r4          ; velocity *= dt
//         vec2add v0, v1          ; position += velocity
//     "#;
    
//     // Lex
//     let mut lexer = Lexer::new(source.to_string());
//     let tokens = lexer.tokenize().expect("Lexer error");
    
//     // Assemble
//     let mut assembler = Assembler::new(tokens);
//     let program = assembler.assemble().expect("Assembly error");
    
//     println!("Program size: {} bits ({} bytes)", 
//              program.bits.len(), 
//              (program.bits.len() + 7) / 8);
//     println!("Float constants: {:?}", program.float_constants);
    
//     // Run
//     let mut vm = VM::new();
//     vm.regs[0] = 100.0;  // position.x
//     vm.regs[1] = 100.0;  // position.y
//     vm.regs[2] = 10.0;   // velocity.x
//     vm.regs[3] = 5.0;    // velocity.y
//     vm.dt = 0.016;       // 60fps
    
//     let result = vm.run(&program);
    
//     println!("Result: {:?}", result);
//     println!("Final position: ({}, {})", vm.regs[0], vm.regs[1]);
//     println!("Final velocity: ({}, {})", vm.regs[2], vm.regs[3]);
// }


// Cargo.toml
/*
[package]
name = "tiny_vm"
version = "0.1.0"
edition = "2021"

[dependencies]
*/

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
    // Vec2Add = 8,
    // Vec2Sub = 9,
    // Vec2Mul = 10,  // Multiply vector by scalar
    // Vec2Length = 11,
    // Vec2Normalize = 12,
    
    // // Math functions
    // Sin = 13,
    // Cos = 14,
    // Sqrt = 15,
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
            //  8 => Ok(Opcode::Vec2Add),
            // 9 => Ok(Opcode::Vec2Sub),
            // 10 => Ok(Opcode::Vec2Mul),
            // 11 => Ok(Opcode::Vec2Length),
            // 12 => Ok(Opcode::Vec2Normalize),
            // 13 => Ok(Opcode::Sin),
            // 14 => Ok(Opcode::Cos),
            // 15 => Ok(Opcode::Sqrt),
            _ => Err(format!("Invalid opcode: {}", value)),
        }
    }
}

// Simple bit container (replace with your packed_bits crate)
struct BitContainer {
    data: Vec<u8>,
    bit_len: usize,
}

impl BitContainer {
    fn new() -> Self {
        Self {
            data: Vec::new(),
            bit_len: 0,
             //float_constants: Vec::new(),
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

struct Program {
    bits: BitContainer,
}

impl Program {
    fn new() -> Self {
        Self {
            bits: BitContainer::new(),
        }
    }

    fn push_mov_imm(&mut self, reg: u8, value: u8) {
        self.bits.push(3, Opcode::MovImm as u32);
        self.bits.push(2, reg as u32);
        self.bits.push(8, value as u32);
    }

    fn push_add(&mut self, ra: u8, rb: u8) {
        self.bits.push(3, Opcode::Add as u32);
        self.bits.push(2, ra as u32);
        self.bits.push(2, rb as u32);
    }

    fn push_sub(&mut self, ra: u8, rb: u8) {
        self.bits.push(3, Opcode::Sub as u32);
        self.bits.push(2, ra as u32);
        self.bits.push(2, rb as u32);
    }

    fn push_mul(&mut self, ra: u8, rb: u8) {
        self.bits.push(3, Opcode::Mul as u32);
        self.bits.push(2, ra as u32);
        self.bits.push(2, rb as u32);
    }
}

#[derive(Debug)]
enum VMResult {
    Complete,
    Kill,
}

struct VM {
    regs: [u32; 4],
}

impl VM {
    fn new() -> Self {
        Self { regs: [0; 4] }
    }

    fn run(&mut self, program: &Program) -> VMResult {
        let mut pc = 0; // bit index

        while pc < program.bits.len() {
            let opcode_bits = match program.bits.get(pc, 3) {
                Some(v) => v,
                None => break,
            };
            pc += 3;

            let opcode = match Opcode::from_u8(opcode_bits as u8) {
                Ok(op) => op,
                Err(_) => break,
            };

            match opcode {
                Opcode::Nop => {}
                
                Opcode::Add => {
                    let ra = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    let rb = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    self.regs[ra] = self.regs[ra].wrapping_add(self.regs[rb]);
                }
                
                Opcode::Sub => {
                    let ra = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    let rb = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    self.regs[ra] = self.regs[ra].wrapping_sub(self.regs[rb]);
                }
                
                Opcode::Mul => {
                    let ra = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    let rb = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    self.regs[ra] = self.regs[ra].wrapping_mul(self.regs[rb]);
                }

                Opcode::Div => {
                    let ra = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    let rb = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    if self.regs[rb] != 0 {
                        self.regs[ra] = self.regs[ra] / self.regs[rb];
                    } else {
                        self.regs[ra] = 0; // Handle division by zero
                    }
                }   
                
                Opcode::MovImm => {
                    let reg = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    let imm = program.bits.get(pc, 8).unwrap();
                    pc += 8;
                    self.regs[reg] = imm;
                }
                
                Opcode::MovReg => {
                    let ra = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    let rb = program.bits.get(pc, 2).unwrap() as usize;
                    pc += 2;
                    self.regs[ra] = self.regs[rb];
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
    println!("=== Tiny VM Demo ===\n");

    // Build a simple program
    let mut program = Program::new();
    program.push_mov_imm(0, 10);  // R0 = 10
    program.push_mov_imm(1, 20);  // R1 = 20
    program.push_add(0, 1);       // R0 += R1 (R0 = 30)
    program.push_mov_imm(2, 5);   // R2 = 5
    program.push_mul(0, 2);       // R0 *= R2 (R0 = 150)

    println!("Program size: {} bits ({} bytes)\n", 
             program.bits.len(),
             (program.bits.len() + 7) / 8);

    // Run it
    let mut vm = VM::new();
    println!("Initial registers: {:?}", vm.regs);
    
    let result = vm.run(&program);
    
    println!("Result: {:?}", result);
    println!("Final registers: {:?}\n", vm.regs);
    
    // Expected: [150, 20, 5, 0]
    assert_eq!(vm.regs[0], 150);
    println!("✓ Test passed!");
}
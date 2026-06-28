/// microgpt.rs — 用纯 Rust 训练和运行 GPT 推理的最小实现
/// 翻译自 @karpathy 的 microgpt.py
///
/// 使用方式:
///   1. 创建 Cargo.toml (见下方) 并将本文件存为 src/main.rs
///   2. cargo run --release
///
/// Cargo.toml:
/// [package]
/// name = "microgpt"
/// version = "0.1.0"
/// edition = "2021"
/// [dependencies]
/// rand = "0.8"

use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::io::{BufRead, BufReader, Write};
use std::rc::Rc;

// ============================================================
// 超参数
// ============================================================
const N_LAYER: usize = 1;     // Transformer 层数/depth (网络深度)
const N_EMBD: usize = 16;     // 嵌入维度/embedding dim (网络宽度)
const BLOCK_SIZE: usize = 16; // 注意力窗口最大上下文长度/max context length (最长名字 15 字符)
const N_HEAD: usize = 4;      // 注意力头数/attention heads
const HEAD_DIM: usize = N_EMBD / N_HEAD; // 每个头的维度/dim per head (= 4)

const LEARNING_RATE: f64 = 0.01; // 学习率/learning rate
const BETA1: f64 = 0.85;         // Adam 一阶动量系数/1st moment
const BETA2: f64 = 0.99;         // Adam 二阶动量系数/2nd moment
const EPS_ADAM: f64 = 1e-8;     // Adam 数值稳定项/numerical stability
const NUM_STEPS: usize = 1000;  // 训练步数/training steps
const TEMPERATURE: f64 = 0.5;   // 采样温度/sampling temp, (0, 1] 低到高=低创造力到高创造力

// ============================================================
// 数据集
// ============================================================
fn load_docs(rng: &mut impl Rng) -> Vec<String> {
    let content = include_str!("../data/names.txt");
    let mut docs: Vec<String> = content
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect();
    docs.shuffle(rng);
    println!("文档数/num docs: {}", docs.len());
    docs
}

// ============================================================
// 分词器
// ============================================================
struct Tokenizer {
    uchars: Vec<char>,
    bos: usize,
    vocab_size: usize,
}

impl Tokenizer {
    fn new(docs: &[String], _rng: &mut impl Rng) -> Self {
        let mut char_set = BTreeSet::new();
        for doc in docs {
            for ch in doc.chars() {
                char_set.insert(ch);
            }
        }
        let uchars: Vec<char> = char_set.into_iter().collect();
        let bos = uchars.len();
        let vocab_size = uchars.len() + 1;
        println!("词表大小/vocab size: {}", vocab_size);
        Tokenizer {
            uchars,
            bos,
            vocab_size,
        }
    }

    fn encode(&self, doc: &str) -> Vec<usize> {
        let mut tokens = vec![self.bos];
        for ch in doc.chars() {
            tokens.push(self.uchars.iter().position(|&c| c == ch).unwrap());
        }
        tokens.push(self.bos);
        tokens
    }
}

// ============================================================
// Autograd: 通过计算图递归应用链式法则
// ============================================================
struct Value {
    data: f64,
    grad: f64,
    children: Vec<ValueRef>,
    local_grads: Vec<f64>,
}

#[derive(Clone)]
struct ValueRef(Rc<RefCell<Value>>);

impl ValueRef {
    fn new(data: f64) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children: vec![],
            local_grads: vec![],
        })))
    }

    fn op(data: f64, children: Vec<ValueRef>, local_grads: Vec<f64>) -> Self {
        ValueRef(Rc::new(RefCell::new(Value {
            data,
            grad: 0.0,
            children,
            local_grads,
        })))
    }

    #[inline]
    fn data(&self) -> f64 {
        self.0.borrow().data
    }

    fn log(&self) -> ValueRef {
        let d = self.data();
        ValueRef::op(d.ln(), vec![self.clone()], vec![1.0 / d])
    }

    fn exp(&self) -> ValueRef {
        let e = self.data().exp();
        ValueRef::op(e, vec![self.clone()], vec![e])
    }

    fn relu(&self) -> ValueRef {
        let d = self.data();
        ValueRef::op(
            d.max(0.0),
            vec![self.clone()],
            vec![if d > 0.0 { 1.0 } else { 0.0 }],
        )
    }

    fn pow(&self, n: f64) -> ValueRef {
        let d = self.data();
        ValueRef::op(d.powf(n), vec![self.clone()], vec![n * d.powf(n - 1.0)])
    }

    /// 反向传播: 迭代式拓扑排序 + 链式法则
    fn backward(&self) {
        let mut topo: Vec<ValueRef> = Vec::new();
        let mut visited = HashSet::new();
        // 迭代式后序遍历 (避免递归栈溢出)
        let mut stack: Vec<(ValueRef, bool)> = vec![(self.clone(), false)];
        while let Some((node, processed)) = stack.pop() {
            let ptr = Rc::as_ptr(&node.0);
            if visited.contains(&ptr) {
                continue;
            }
            if processed {
                visited.insert(ptr);
                topo.push(node);
                continue;
            }
            stack.push((node.clone(), true));
            let children: Vec<ValueRef> = node.0.borrow().children.clone();
            for child in children {
                let cptr = Rc::as_ptr(&child.0);
                if !visited.contains(&cptr) {
                    stack.push((child, false));
                }
            }
        }
        // 设置输出梯度为 1, 然后反向传播
        self.0.borrow_mut().grad = 1.0;
        for node in topo.iter().rev() {
            let b = node.0.borrow();
            let g = b.grad;
            let pairs: Vec<(ValueRef, f64)> = b
                .children
                .iter()
                .zip(b.local_grads.iter())
                .map(|(c, &lg)| (c.clone(), lg))
                .collect();
            drop(b);
            for (child, lg) in pairs {
                child.0.borrow_mut().grad += lg * g;
            }
        }
    }
}

// --- 运算符重载 ---
impl std::ops::Add for ValueRef {
    type Output = ValueRef;
    fn add(self, rhs: Self) -> ValueRef {
        let d = self.data() + rhs.data();
        ValueRef::op(d, vec![self, rhs], vec![1.0, 1.0])
    }
}

impl std::ops::Mul for ValueRef {
    type Output = ValueRef;
    fn mul(self, rhs: Self) -> ValueRef {
        let (a, b) = (self.data(), rhs.data());
        ValueRef::op(a * b, vec![self, rhs], vec![b, a])
    }
}

impl std::ops::Neg for ValueRef {
    type Output = ValueRef;
    fn neg(self) -> ValueRef {
        let d = self.data();
        ValueRef::op(-d, vec![self], vec![-1.0])
    }
}

impl std::ops::Sub for ValueRef {
    type Output = ValueRef;
    fn sub(self, rhs: Self) -> ValueRef {
        self + (-rhs)
    }
}

impl std::ops::Div for ValueRef {
    type Output = ValueRef;
    fn div(self, rhs: Self) -> ValueRef {
        self * rhs.pow(-1.0)
    }
}

// ============================================================
// 辅助函数
// ============================================================
type Matrix = Vec<Vec<ValueRef>>;

/// Box-Muller 变换: 标准正态分布采样
fn randn(rng: &mut impl Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-15);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn make_matrix(nout: usize, nin: usize, std: f64, rng: &mut impl Rng) -> Matrix {
    (0..nout)
        .map(|_| (0..nin).map(|_| ValueRef::new(randn(rng) * std)).collect())
        .collect()
}

/// 线性层: y = W @ x
fn linear(x: &[ValueRef], w: &Matrix) -> Vec<ValueRef> {
    w.iter()
        .map(|row| {
            let zero = ValueRef::new(0.0);
            row.iter()
                .zip(x.iter())
                .fold(zero, |acc, (wi, xi)| acc + wi.clone() * xi.clone())
        })
        .collect()
}

/// Softmax 归一化
fn softmax(logits: &[ValueRef]) -> Vec<ValueRef> {
    let max_val = logits
        .iter()
        .map(|v| v.data())
        .fold(f64::NEG_INFINITY, f64::max);
    // 关键: max_node 必须是 Value 节点, 这样 (logit - max) 才能保持计算图连通
    // 否则梯度会在 shifted 处断裂, 无法回传到 logits (即模型参数)
    let max_node = ValueRef::new(max_val);
    let exps: Vec<ValueRef> = logits
        .iter()
        .map(|val| (val.clone() - max_node.clone()).exp())
        .collect();
    let total = exps
        .iter()
        .fold(ValueRef::new(0.0), |acc, e| acc + e.clone());
    exps.iter().map(|e| e.clone() / total.clone()).collect()
}

/// RMS 归一化
fn rmsnorm(x: &[ValueRef]) -> Vec<ValueRef> {
    let n = x.len() as f64;
    let ms: f64 = x.iter().map(|xi| {
        let d = xi.data();
        d * d
    }).sum::<f64>() / n;
    let scale = ValueRef::new((ms + 1e-5).powf(-0.5));
    x.iter().map(|xi| xi.clone() * scale.clone()).collect()
}

// ============================================================
// GPT 前向传播
// ============================================================
fn gpt(
    token_id: usize,
    pos_id: usize,
    keys: &mut [Vec<Vec<ValueRef>>],
    values: &mut [Vec<Vec<ValueRef>>],
    sd: &HashMap<String, Matrix>,
) -> Vec<ValueRef> {
    let tok_emb = &sd["wte"][token_id];
    let pos_emb = &sd["wpe"][pos_id];
    let mut x: Vec<ValueRef> = tok_emb
        .iter()
        .zip(pos_emb.iter())
        .map(|(t, p)| t.clone() + p.clone())
        .collect();
    x = rmsnorm(&x);

    for li in 0..N_LAYER {
        // 1) 多头注意力
        let x_res = x.clone();
        x = rmsnorm(&x);
        let q = linear(&x, &sd[&format!("layer{li}.attn_wq")]);
        let k = linear(&x, &sd[&format!("layer{li}.attn_wk")]);
        let v = linear(&x, &sd[&format!("layer{li}.attn_wv")]);
        keys[li].push(k);
        values[li].push(v);

        let mut x_attn = Vec::with_capacity(N_EMBD);
        for h in 0..N_HEAD {
            let hs = h * HEAD_DIM;
            let q_h: Vec<ValueRef> = q[hs..hs + HEAD_DIM].to_vec();
            let k_h: Vec<Vec<ValueRef>> = keys[li]
                .iter()
                .map(|ki| ki[hs..hs + HEAD_DIM].to_vec())
                .collect();
            let v_h: Vec<Vec<ValueRef>> = values[li]
                .iter()
                .map(|vi| vi[hs..hs + HEAD_DIM].to_vec())
                .collect();
            let t_len = k_h.len();

            let scale = ValueRef::new(1.0 / (HEAD_DIM as f64).sqrt());
            let attn_logits: Vec<ValueRef> = (0..t_len)
                .map(|t| {
                    let zero = ValueRef::new(0.0);
                    let dot = q_h.iter().zip(k_h[t].iter()).fold(zero, |acc, (qj, kj)| {
                        acc + qj.clone() * kj.clone()
                    });
                    dot * scale.clone()
                })
                .collect();

            let attn_weights = softmax(&attn_logits);
            let head_out: Vec<ValueRef> = (0..HEAD_DIM)
                .map(|j| {
                    let zero = ValueRef::new(0.0);
                    (0..t_len).fold(zero, |acc, t| {
                        acc + attn_weights[t].clone() * v_h[t][j].clone()
                    })
                })
                .collect();
            x_attn.extend(head_out);
        }

        x = linear(&x_attn, &sd[&format!("layer{li}.attn_wo")]);
        x = x
            .iter()
            .zip(x_res.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();

        // 2) MLP 块
        let x_res = x.clone();
        x = rmsnorm(&x);
        x = linear(&x, &sd[&format!("layer{li}.mlp_fc1")]);
        x = x.iter().map(|xi| xi.relu()).collect();
        x = linear(&x, &sd[&format!("layer{li}.mlp_fc2")]);
        x = x
            .iter()
            .zip(x_res.iter())
            .map(|(a, b)| a.clone() + b.clone())
            .collect();
    }

    linear(&x, &sd["lm_head"])
}

// ============================================================
// 模型参数: 初始化 / 保存 / 加载
// ============================================================
fn init_params(vocab_size: usize, rng: &mut impl Rng) -> HashMap<String, Matrix> {
    let mut sd: HashMap<String, Matrix> = HashMap::new();
    sd.insert("wte".into(), make_matrix(vocab_size, N_EMBD, 0.08, rng));
    sd.insert("wpe".into(), make_matrix(BLOCK_SIZE, N_EMBD, 0.08, rng));
    sd.insert("lm_head".into(), make_matrix(vocab_size, N_EMBD, 0.08, rng));
    for i in 0..N_LAYER {
        sd.insert(format!("layer{i}.attn_wq"), make_matrix(N_EMBD, N_EMBD, 0.08, rng));
        sd.insert(format!("layer{i}.attn_wk"), make_matrix(N_EMBD, N_EMBD, 0.08, rng));
        sd.insert(format!("layer{i}.attn_wv"), make_matrix(N_EMBD, N_EMBD, 0.08, rng));
        sd.insert(format!("layer{i}.attn_wo"), make_matrix(N_EMBD, N_EMBD, 0.08, rng));
        sd.insert(format!("layer{i}.mlp_fc1"), make_matrix(4 * N_EMBD, N_EMBD, 0.08, rng));
        sd.insert(format!("layer{i}.mlp_fc2"), make_matrix(N_EMBD, 4 * N_EMBD, 0.08, rng));
    }
    sd
}

/// 将模型参数保存为文本文件 (每行: 参数名 + 空格分隔的浮点值)
fn save_model(sd: &HashMap<String, Matrix>, path: &str) {
    let mut file = std::fs::File::create(path).unwrap();
    let mut names: Vec<&String> = sd.keys().collect();
    names.sort();
    let mut total_params = 0usize;
    for name in &names {
        let mat = &sd[*name];
        let count: usize = mat.iter().map(|row| row.len()).sum();
        total_params += count;
        let flat: Vec<String> = mat
            .iter()
            .flat_map(|row| row.iter().map(|v| format!("{:.17e}", v.data())))
            .collect();
        writeln!(file, "{} {}", name, flat.join(" ")).unwrap();
    }
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    println!("\n模型已保存/model saved to {}", path);
    println!("  参数总量/params:     {}", total_params);
    println!("  层数/layers:         {}", N_LAYER);
    println!("  嵌入维度/embd:       {}", N_EMBD);
    println!("  注意力头/heads:      {} (dim/每头维度={})", N_HEAD, HEAD_DIM);
    println!("  上下文长度/block:    {}", BLOCK_SIZE);
    println!("  文件大小/file size:  {:.1} KB", file_size as f64 / 1024.0);
}

/// 从文本文件加载模型参数
fn load_model(path: &str, vocab_size: usize) -> HashMap<String, Matrix> {
    let file = std::fs::File::open(path).unwrap_or_else(|_| {
        eprintln!("错误: 模型文件 '{}' 不存在", path);
        eprintln!("请先运行训练: cargo run --release -- train {}", path);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);
    let mut sd: HashMap<String, Matrix> = HashMap::new();
    for line in reader.lines() {
        let line = line.unwrap();
        let mut parts = line.splitn(2, ' ');
        let name = parts.next().unwrap().to_string();
        let flat: Vec<f64> = parts
            .next()
            .unwrap_or("")
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        let (rows, cols) = match name.as_str() {
            "wte" | "lm_head" => (vocab_size, N_EMBD),
            "wpe" => (BLOCK_SIZE, N_EMBD),
            _ if name.contains("mlp_fc1") => (4 * N_EMBD, N_EMBD),
            _ if name.contains("mlp_fc2") => (N_EMBD, 4 * N_EMBD),
            _ => (N_EMBD, N_EMBD),
        };
        let matrix: Matrix = (0..rows)
            .map(|r| (0..cols).map(|c| ValueRef::new(flat[r * cols + c])).collect())
            .collect();
        sd.insert(name, matrix);
    }
    println!("模型已加载/model loaded from {}", path);
    sd
}

/// 推理: 生成新名字
fn run_inference(sd: &HashMap<String, Matrix>, tok: &Tokenizer, rng: &mut impl Rng) {
    println!("\n--- 推理/inference (生成新名字/hallucinated names) ---");
    for sample_idx in 0..20 {
        let mut keys: Vec<Vec<Vec<ValueRef>>> = vec![vec![]; N_LAYER];
        let mut vals: Vec<Vec<Vec<ValueRef>>> = vec![vec![]; N_LAYER];
        let mut token_id = tok.bos;
        let mut sample = String::new();
        for pos_id in 0..BLOCK_SIZE {
            let logits = gpt(token_id, pos_id, &mut keys, &mut vals, sd);
            let scaled: Vec<ValueRef> = logits
                .iter()
                .map(|l| l.clone() * ValueRef::new(1.0 / TEMPERATURE))
                .collect();
            let probs = softmax(&scaled);
            let weights: Vec<f64> = probs.iter().map(|p| p.data()).collect();
            token_id = weighted_choice(&weights, rng);
            if token_id == tok.bos {
                break;
            }
            sample.push(tok.uchars[token_id]);
        }
        println!("sample {:2}: {}", sample_idx + 1, sample);
    }
}

// ============================================================
// 主函数
// ============================================================
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("");
    let model_path = args.get(2).map(|s| s.as_str()).unwrap_or("model.weights");

    // Let there be order among chaos
    let mut rng = StdRng::seed_from_u64(42);

    // 数据
    let docs = load_docs(&mut rng);
    let tok = Tokenizer::new(&docs, &mut rng);

    match mode {
        "train" => {
            let sd = init_params(tok.vocab_size, &mut rng);
            train_model(&sd, &docs, &tok);
            save_model(&sd, model_path);
        }
        "infer" => {
            let sd = load_model(model_path, tok.vocab_size);
            let mut infer_rng = StdRng::seed_from_u64(1337);
            run_inference(&sd, &tok, &mut infer_rng);
        }
        _ => {
            // 默认: 训练 + 保存 + 推理
            let sd = init_params(tok.vocab_size, &mut rng);
            println!("参数总量/num params: {}", count_params(&sd));
            train_model(&sd, &docs, &tok);
            save_model(&sd, model_path);
            let mut infer_rng = StdRng::seed_from_u64(1337);
            run_inference(&sd, &tok, &mut infer_rng);
        }
    }
}

fn count_params(sd: &HashMap<String, Matrix>) -> usize {
    sd.values().map(|mat| mat.iter().map(|row| row.len()).sum::<usize>()).sum()
}

fn train_model(sd: &HashMap<String, Matrix>, docs: &[String], tok: &Tokenizer) {
    let params: Vec<ValueRef> = sd
        .values()
        .flat_map(|mat| mat.iter().flat_map(|row| row.iter().cloned()))
        .collect();
    let n_params = params.len();
    let mut adam_m = vec![0.0f64; n_params];
    let mut adam_v = vec![0.0f64; n_params];

    for step in 0..NUM_STEPS {
        let doc = &docs[step % docs.len()];
        let tokens = tok.encode(doc);
        let n = BLOCK_SIZE.min(tokens.len() - 1);

        let mut keys: Vec<Vec<Vec<ValueRef>>> = vec![vec![]; N_LAYER];
        let mut vals: Vec<Vec<Vec<ValueRef>>> = vec![vec![]; N_LAYER];
        let mut losses: Vec<ValueRef> = Vec::with_capacity(n);

        for pos_id in 0..n {
            let token_id = tokens[pos_id];
            let target_id = tokens[pos_id + 1];
            let logits = gpt(token_id, pos_id, &mut keys, &mut vals, sd);
            let probs = softmax(&logits);
            let loss_t = -probs[target_id].log();
            losses.push(loss_t);
        }

        let n_inv = ValueRef::new(1.0 / n as f64);
        let loss = losses
            .into_iter()
            .fold(ValueRef::new(0.0), |acc, l| acc + l)
            * n_inv;

        loss.backward();

        let lr_t = LEARNING_RATE * (1.0 - step as f64 / NUM_STEPS as f64);
        for (i, p) in params.iter().enumerate() {
            let mut pb = p.0.borrow_mut();
            let g = pb.grad;
            adam_m[i] = BETA1 * adam_m[i] + (1.0 - BETA1) * g;
            adam_v[i] = BETA2 * adam_v[i] + (1.0 - BETA2) * g * g;
            let m_hat = adam_m[i] / (1.0 - BETA1.powi((step + 1) as i32));
            let v_hat = adam_v[i] / (1.0 - BETA2.powi((step + 1) as i32));
            pb.data -= lr_t * m_hat / (v_hat.sqrt() + EPS_ADAM);
            pb.grad = 0.0;
        }

        print!(
            "\rstep {:4} / {:4} | loss {:.4}",
            step + 1,
            NUM_STEPS,
            loss.data()
        );
        std::io::stdout().flush().unwrap();
    }
}

/// 按权重随机采样
fn weighted_choice(weights: &[f64], rng: &mut impl Rng) -> usize {
    let total: f64 = weights.iter().sum();
    let mut r = rng.gen::<f64>() * total;
    for (i, &w) in weights.iter().enumerate() {
        r -= w;
        if r <= 0.0 {
            return i;
        }
    }
    weights.len() - 1
}

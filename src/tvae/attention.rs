/*
class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super(SelfAttentionLayer, self).__init__()
        self.feature_size = feature_size

        # Linear transformations for Q, K, V from the same source
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        # Apply linear transformations
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        # Scaled dot-product attention
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.feature_size, dtype=torch.float32))

        # Apply mask (if provided)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Multiply weights with values
        output = torch.matmul(attention_weights, values)

        return output, attention_weights

*/

// https://stackoverflow.com/questions/76648620/how-do-i-implement-this-attention-layer-in-pytorch

use tch::{nn::{self, Module}, Device, Kind, Tensor};

#[derive(Debug)]
pub struct SelfAttentionLayer {
    key: nn::Linear,
    query: nn::Linear,
    value: nn::Linear,
    // out_proj: nn::Linear,
    aligned_input_dim: i64,
    seq_len: i64,
    unit_len: i64,
}

// const HIDDEN_DIM: i64 = 4;

pub fn self_attention(vs: nn::Path, input_dim: i64) -> SelfAttentionLayer {
    // let hidden_size = input_dim * HIDDEN_DIM;

    // let key = nn::linear(&vs / "key", input_dim, hidden_size, Default::default());
    // let query = nn::linear(&vs / "query", input_dim, hidden_size, Default::default());
    // let value = nn::linear(&vs / "value", input_dim, hidden_size, Default::default());
    // let out_proj = nn::linear(&vs / "out", hidden_size, input_dim, Default::default());

    let seq_len = (input_dim as f64).sqrt().ceil() as i64;
    // let seq_len = input_dim;
    let unit_len = (input_dim as f64 / seq_len as f64).ceil() as i64;
    // let unit_len = 2;
    let aligned_input_dim = seq_len * unit_len;

    let key = nn::linear(&vs / "key", unit_len, unit_len, Default::default());
    let query = nn::linear(&vs / "query", unit_len, unit_len, Default::default());
    let value = nn::linear(&vs / "value", unit_len, unit_len, Default::default());

    SelfAttentionLayer {
        key,
        query,
        value,
        // out_proj,
        aligned_input_dim,
        seq_len,
        unit_len
    }
}

impl Module for SelfAttentionLayer {
    fn forward(&self, xs: &tch::Tensor) -> tch::Tensor {
        
        // let v0 = xs.eq_tensor(&Tensor::zeros(1, (Kind::Int8, Device::Cpu))).to_kind(Kind::Float);
        // let v1 = xs.eq_tensor(&Tensor::ones(1, (Kind::Int8, Device::Cpu))).to_kind(Kind::Float);

        // let xs = Tensor::stack(&[v0, v1], 2);
        // println!("xs {:?}", xs.size());

        // let xs_size = xs.size();
        // let att_shape = [xs_size[0], HIDDEN_DIM, xs_size[1]];

        // // println!("xs {:?}", xs.size());
        // let key = self.key.forward(xs).reshape(att_shape);
        // let query = self.query.forward(xs).reshape(att_shape);
        // let value = self.value.forward(xs).reshape(att_shape);
        // // println!("key {:?}", key.size());

        // let scores = query.bmm(&key.transpose(1, 2)) / (self.input_dim as f64).sqrt();
        // let attention = scores.softmax(-1, None);

        // let weighted = attention.bmm(&value);
        // // println!("attw {:?}", weighted.size());

        // let weighted_flat = weighted.reshape([xs_size[0], xs_size[1] * HIDDEN_DIM]);

        // self.out_proj.forward(&weighted_flat)

        let xs_size = xs.size();
        let att_shape = [xs_size[0], self.seq_len, self.unit_len];

        let pad_num = self.aligned_input_dim - xs_size[1];
        // println!("{} {} {}", pad_num, self.aligned_input_dim, xs_size[1]);
        let xs = xs.pad([0, pad_num], "constant", 0.0).reshape(att_shape);
        // println!("xs {:?}", xs.size());

        // println!("xs {:?}", xs.size());
        let key = self.key.forward(&xs);//.reshape(att_shape);
        let query = self.query.forward(&xs);//.reshape(att_shape);
        let value = self.value.forward(&xs);//.reshape(att_shape);
        // println!("key {:?}", key.size());

        let scores = query.bmm(&key.transpose(1, 2)) / (self.unit_len as f64).sqrt();
        let attention = scores.softmax(-1, None);

        let weighted = attention.bmm(&value);
        // println!("attw {:?}", weighted.size());

        let weighted_flat = weighted.reshape([xs_size[0], self.aligned_input_dim]);

        weighted_flat.slice(1, 0, xs_size[1], 1)
    }
}

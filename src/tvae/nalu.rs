use tch::nn;
use tch::nn::Module;
use tch::Tensor;

// fn xavier_init(fan_in: i64, fan_out: i64) -> nn::Init {
//     let gain = 1.0;
//     let bound = 6.0 / (fan_in as f64 + fan_out as f64).sqrt();
//     nn::Init::Uniform {
//         lo: -bound,
//         up: bound,
//     }
// }

#[derive(Debug)]
pub struct NacCell {
    w: Tensor,
    m: Tensor,
}

impl NacCell {
    pub fn new(vs: nn::Path, in_shape: i64, out_shape: i64) -> Self {
        let w = vs.entry("W").or_kaiming_uniform(&[out_shape, in_shape]);
        let m = vs.entry("M").or_kaiming_uniform(&[out_shape, in_shape]);
        Self { w, m }
    }
}

impl Module for NacCell {
    fn forward(&self, xs: &Tensor) -> Tensor {
        let w = self.w.tanh() * self.m.sigmoid();
        xs.linear::<Tensor>(&w, None)
    }
}

#[derive(Debug)]
pub struct NaluCell {
    nac: NacCell,
    g: Tensor,
}

impl NaluCell {
    pub fn new(vs: nn::Path, in_shape: i64, out_shape: i64) -> Self {
        let nac = NacCell::new(&vs / "nac", in_shape, out_shape);
        let g = vs.entry("G").or_kaiming_uniform(&[out_shape, in_shape]);

        /*

            class NaluCell(nn.Module):
        """Basic NALU unit implementation
        from https://arxiv.org/pdf/1808.00508.pdf
        """

        def __init__(self, in_shape, out_shape):
            """
            in_shape: input sample dimension
            out_shape: output sample dimension
            """
            super().__init__()
            self.in_shape = in_shape
            self.out_shape = out_shape
            self.G = Parameter(Tensor(out_shape, in_shape))
            self.nac = NacCell(out_shape, in_shape)
            xavier_uniform_(self.G)
            self.eps = 1e-5
            self.register_parameter('bias', None)

        def forward(self, input):
            a = self.nac(input)
            g = sigmoid(linear(input, self.G, self.bias))
            ag = g * a
            log_in = log(abs(input) + self.eps)
            m = exp(self.nac(log_in))
            md = (1 - g) * m
            return ag + md

             */

        Self { nac, g }
    }
}

impl Module for NaluCell {
    fn forward(&self, xs: &Tensor) -> Tensor {
        const EPS: f64 = 1e-5;

        let a = self.nac.forward(xs);
        let g = xs.linear::<Tensor>(&self.g, None).sigmoid();

        let ag = &g * &a;
        let log_in = (xs.abs() + EPS).log();

        let m = self.nac.forward(&log_in);
        let md = (1.0 - g) * m;

        ag + md
    }
}

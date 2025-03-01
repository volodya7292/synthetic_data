use rand::{seq::SliceRandom, SeedableRng};

#[derive(Clone, Copy)]
pub enum ColumnDataRef<'a> {
    Discrete(&'a [i32]),
    Continuous(&'a [f32]),
}

impl ColumnDataRef<'_> {
    pub fn len(&self) -> usize {
        match self {
            Self::Discrete(data) => data.len(),
            Self::Continuous(data) => data.len(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ColumnData {
    Discrete(Vec<i32>),
    Continuous(Vec<f32>),
}

impl ColumnData {
    pub fn element_size(&self) -> usize {
        match self {
            Self::Discrete(_) => 4,
            Self::Continuous(_) => 4,
        }
    }

    pub fn as_ref(&self) -> ColumnDataRef {
        match self {
            ColumnData::Discrete(v) => ColumnDataRef::Discrete(v),
            ColumnData::Continuous(v) => ColumnDataRef::Continuous(v),
        }
    }

    /// Prediclable shuffling
    pub fn pshuffle(&mut self) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(123);
        match self {
            ColumnData::Discrete(v) => v.shuffle(&mut rng),
            ColumnData::Continuous(v) => v.shuffle(&mut rng),
        }
    }

    pub fn take_train_part(&self, p: f32) -> ColumnDataRef {
        assert!((0.0..=1.0).contains(&p));

        let slice_len = (self.as_ref().len() as f32 * p) as usize;

        match self {
            ColumnData::Discrete(v) => ColumnDataRef::Discrete(&v[..slice_len]),
            ColumnData::Continuous(v) => ColumnDataRef::Continuous(&v[..slice_len]),
        }
    }

    pub fn take_test_part(&self, p: f32) -> ColumnDataRef {
        assert!((0.0..=1.0).contains(&p));

        let slice_len = (self.as_ref().len() as f32 * p) as usize;

        match self {
            ColumnData::Discrete(v) => ColumnDataRef::Discrete(&v[slice_len..]),
            ColumnData::Continuous(v) => ColumnDataRef::Continuous(&v[slice_len..]),
        }
    }
}

pub type Realness = f32;

#[derive(Debug)]
pub enum SampledColumnData {
    Discrete(Vec<i32>, Realness),
    Continuous(Vec<f32>, Realness),
}

impl SampledColumnData {
    pub(crate) fn from_regular(regular: ColumnData, realness: f32) -> Self {
        match regular {
            ColumnData::Discrete(data) => Self::Discrete(data, realness),
            ColumnData::Continuous(data) => Self::Continuous(data, realness),
        }
    }

    pub fn realness(&self) -> Realness {
        match self {
            Self::Discrete(_, v) | Self::Continuous(_, v) => *v,
        }
    }

    pub fn data_as_ref(&self) -> ColumnDataRef {
        match self {
            Self::Discrete(data, ..) => ColumnDataRef::Discrete(data),
            Self::Continuous(data, ..) => ColumnDataRef::Continuous(data),
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            Self::Discrete(..) => 4,
            Self::Continuous(..) => 4,
        }
    }
}

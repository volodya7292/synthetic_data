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

#[derive(Debug)]
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

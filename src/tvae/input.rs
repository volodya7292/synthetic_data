#[derive(Debug)]
pub enum ColumnData {
    Discrete(Vec<i32>),
    Continuous(Vec<f32>),
}

impl ColumnData {
    pub fn element_size(&self) -> usize {
        match self {
            ColumnData::Discrete(_) => 4,
            ColumnData::Continuous(_) => 4,
        }
    }
}

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

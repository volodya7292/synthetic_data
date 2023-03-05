pub enum ColumnData {
    Discrete(Vec<i32>),
    Continuous(Vec<f32>),
}

impl ColumnData {
    pub fn len(&self) -> usize {
        match self {
            ColumnData::Discrete(data) => data.len(),
            ColumnData::Continuous(data) => data.len(),
        }
    }
}

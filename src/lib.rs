use std::vec::Vec;

/// The normalized data is stored in the 2D Matrix (Vector of Vectors) as follow:
///
///         col1    col2    col3 ...
///    row1    x       x       x
///    row2    x       x       x
///    row3    x       x       x
///    ...
///
/// Each column describes a different feature of some dataset. Each row is an
/// entry in the dataset. There can be as many rows and as many columns, so do
/// not hardcode these indices.
///
/// The labels vector contains labels that correspond to the data on each row.
/// This means labels[i] is associated with the data in row[i].
pub struct Dataset {
    pub matrix: Vec<Vec<f64>>,
    pub labels: Vec<String>,
}

impl Dataset {
    /// @param input - a vector of N pieces of data
    /// @return a String indicating the label of the piece of data closest in
    ///         distance to the given input
    pub fn predict(&self, input: &Vec<f64>) -> String {
        let data : Vec<Vec<f64>> = self.matrix.clone();
        let label_points: Vec<String> = self.labels.clone();
        let mut ans = "".to_string();
        let mut min: f64 = 1000000.0;
        let mut count : usize = 0;
        for points in data.iter() {
            let dist = distance(input, points);
            if dist < min {
                min = dist;
                ans = label_points[count].clone();
            }
            count += 1;
        }
        ans
    }
}


/// @param a - an N-dimentional vector of doubles
/// @param b - an N-dimentional vector of doubles
/// @return the cartesian distance between the given vectors
pub fn distance(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .fold(0.0f64, |sum, (x, y)| sum + (x - y).powf(2.0))
        .sqrt()
}


/// @param data - a 2D Matrix that will be normalized by the COLUMN
/// @param means - a Vector containing the mean of each COLUMN
/// @param stds - a Vector containing the standard deviation of each COLUMN
pub fn normalize(data: &mut Vec<Vec<f64>>, means: &Vec<f64>, stds: &Vec<f64>) {
    let mut count : usize = 0;
    for points1 in data.iter_mut() {
        count = 0;
        for points2 in points1.iter_mut() {
            *points2 = (*points2 - means[count]) / stds[count];
            count += 1;
        }
    }
}


/// @param data - a 2D Matrix of data
/// @return a vector that contains the mean of each COLUMN in the given dataset.
pub fn mean(data: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut means : Vec<f64> = Vec::new();
    let mut count : f64 = 0.0;
    let mut avg : f64 = 0.0;
    let mut sum : f64 = 0.0;
    for i in 0..data[0].len() {
        count = 0.0;
        sum = 0.0;
        avg = 0.0;
        for j in 0..data.len() {
            sum += data[j][i];
            count += 1.0;
        }
        avg = sum / count;
        means.push(avg);
    }
    means
}


/// @param data - a 2D Matrix of data
/// @return a vector that contains the standard deviation of each COLUMN in the
///         given dataset.
pub fn std(data: &Vec<Vec<f64>>, means: &Vec<f64>) -> Vec<f64> {
    let mut stds : Vec<f64> = Vec::new();
    for i in 0..data[0].len() {
        let mut count : f64 = 0.0;
        let mut sum : f64 = 0.0;
        let mut stnd = 0.0;
        for j in 0..data.len() {
            let mut col_sums = data[j][i] - means[i];
            col_sums = col_sums.powf(2.0);
            sum += col_sums;
            count += 1.0;
        }
        stnd = sum / count;
        stnd = stnd.sqrt();
        stds.push(stnd);
    }
    stds
}

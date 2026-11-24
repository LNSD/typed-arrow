//! A union whose variants are `#[derive(Record)]` structs, read back through views.
//!
//! Regression: the generated union view enum derived `Debug` and `Clone`, which record, list and
//! map views do not implement, so this did not compile with the `views` feature.

use typed_arrow::{List, arrow_array::RecordBatch, prelude::*};

#[derive(Record, Debug, Clone, PartialEq)]
pub struct WithList {
    xs: List<u64>,
}

#[derive(Record, Debug, Clone, PartialEq)]
pub struct WithNumber {
    n: u64,
}

#[derive(Union, Debug, Clone, PartialEq)]
#[union(mode = "dense")]
pub enum Payload {
    #[union(tag = 1, field = "a")]
    A(WithList),
    #[union(tag = 2, field = "b")]
    B(WithNumber),
}

#[derive(Record, Debug, Clone, PartialEq)]
pub struct Row {
    payload: Payload,
}

#[test]
fn union_of_records_round_trips_through_views() -> Result<(), SchemaError> {
    let rows = vec![
        Row {
            payload: Payload::A(WithList {
                xs: List::new(vec![1, 2, 3]),
            }),
        },
        Row {
            payload: Payload::B(WithNumber { n: 7 }),
        },
        Row {
            payload: Payload::A(WithList {
                xs: List::new(vec![]),
            }),
        },
    ];

    let mut builders = <Row as BuildRows>::new_builders(rows.len());
    builders.append_rows(rows.clone());
    let batch: RecordBatch = builders.finish().into_record_batch();

    let mut decoded = Vec::new();
    for view in batch.iter_views::<Row>()? {
        decoded.push(Row::try_from(view?)?);
    }
    assert_eq!(decoded, rows);
    Ok(())
}

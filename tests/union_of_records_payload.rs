//! A payload described entirely with derives: a dense union of record structs, one empty and one
//! holding an `OrderedMap`, nested in a record and round-tripped through a `RecordBatch`.

use std::collections::BTreeMap;

use typed_arrow::{List, OrderedMap, arrow_array::RecordBatch, prelude::*};

#[derive(Record, Debug, Clone, PartialEq)]
pub struct Heartbeat {}

#[derive(Record, Debug, Clone, PartialEq)]
pub struct Counters {
    values: OrderedMap<String, u64>,
}

#[derive(Record, Debug, Clone, PartialEq)]
pub struct Samples {
    xs: List<u64>,
}

#[derive(Union, Debug, Clone, PartialEq)]
#[union(mode = "dense")]
pub enum Message {
    #[union(tag = 1, field = "heartbeat")]
    Heartbeat(Heartbeat),
    #[union(tag = 2, field = "counters")]
    Counters(Counters),
    #[union(tag = 3, field = "samples")]
    Samples(Samples),
}

#[derive(Record, Debug, Clone, PartialEq)]
pub struct Envelope {
    seq: u64,
    message: Message,
}

#[test]
fn union_payload_round_trips_through_record_batch() -> Result<(), SchemaError> {
    let rows = vec![
        Envelope {
            seq: 1,
            message: Message::Heartbeat(Heartbeat {}),
        },
        Envelope {
            seq: 2,
            message: Message::Counters(Counters {
                values: OrderedMap::new(BTreeMap::from([
                    ("errors".to_string(), 3),
                    ("requests".to_string(), 120),
                ])),
            }),
        },
        Envelope {
            seq: 3,
            message: Message::Samples(Samples {
                xs: List::new(vec![4, 5, 6]),
            }),
        },
        Envelope {
            seq: 4,
            message: Message::Heartbeat(Heartbeat {}),
        },
    ];

    let mut builders = <Envelope as BuildRows>::new_builders(rows.len());
    builders.append_rows(rows.clone());
    let batch: RecordBatch = builders.finish().into_record_batch();
    assert_eq!(batch.num_rows(), rows.len());

    let mut decoded = Vec::new();
    for view in batch.iter_views::<Envelope>()? {
        decoded.push(Envelope::try_from(view?)?);
    }
    assert_eq!(decoded, rows);
    Ok(())
}

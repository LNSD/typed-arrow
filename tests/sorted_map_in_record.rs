//! Maps that declare `keys_sorted = true`, nested in a `#[derive(Record)]` struct.
//!
//! Regression: the struct builder finished the nested map through `ArrayBuilder`, which produced
//! an unsorted map, and `StructBuilder::finish` panicked on the type mismatch with the field.

use std::collections::BTreeMap;

use typed_arrow::{
    Map, OrderedMap,
    arrow_array::{Array, RecordBatch, StructArray},
    arrow_schema::DataType,
    bridge::{ArrowBinding, ArrowBindingView},
    prelude::*,
};

#[derive(Record, Debug, Clone, PartialEq)]
pub struct WithOrderedMap {
    m: OrderedMap<String, u64>,
    nullable_values: OrderedMap<String, Option<u64>>,
}

#[derive(Record, Debug, Clone, PartialEq)]
pub struct WithSortedMap {
    m: Map<String, u64, true>,
    nullable_values: Map<String, Option<u64>, true>,
}

fn keys_sorted(data_type: &DataType) -> bool {
    match data_type {
        DataType::Map(_, sorted) => *sorted,
        other => panic!("expected a map, got {other:?}"),
    }
}

fn ordered_row() -> WithOrderedMap {
    WithOrderedMap {
        m: OrderedMap::new(BTreeMap::from([("a".to_string(), 1), ("b".to_string(), 2)])),
        nullable_values: OrderedMap::new(BTreeMap::from([
            ("x".to_string(), Some(3)),
            ("y".to_string(), None),
        ])),
    }
}

#[test]
fn ordered_map_in_record_round_trips_through_views() -> Result<(), SchemaError> {
    let row = ordered_row();
    let mut builder = <WithOrderedMap as ArrowBinding>::new_builder(1);
    <WithOrderedMap as ArrowBinding>::append_value(&mut builder, &row);
    let array: StructArray = <WithOrderedMap as ArrowBinding>::finish(builder);
    assert!(keys_sorted(array.column(0).data_type()));
    assert!(keys_sorted(array.column(1).data_type()));

    let view = <WithOrderedMap as ArrowBindingView>::get_view(&array, 0)?;
    assert_eq!(WithOrderedMap::try_from(view)?, row);
    Ok(())
}

#[test]
fn sorted_map_in_record_round_trips_through_record_batch() -> Result<(), SchemaError> {
    let rows = vec![
        WithSortedMap {
            m: Map::new(vec![("a".to_string(), 1), ("b".to_string(), 2)]),
            nullable_values: Map::new(vec![("x".to_string(), None)]),
        },
        WithSortedMap {
            m: Map::new(vec![]),
            nullable_values: Map::new(vec![("y".to_string(), Some(4))]),
        },
    ];
    let mut builders = <WithSortedMap as BuildRows>::new_builders(rows.len());
    builders.append_rows(rows.clone());
    let batch: RecordBatch = builders.finish().into_record_batch();
    assert!(keys_sorted(batch.column(0).data_type()));
    assert!(keys_sorted(batch.column(1).data_type()));

    let mut decoded = Vec::new();
    for view in batch.iter_views::<WithSortedMap>()? {
        decoded.push(WithSortedMap::try_from(view?)?);
    }
    assert_eq!(decoded, rows);
    Ok(())
}

#[test]
fn standalone_sorted_map_declares_sorted_keys() {
    let mut builder = <Map<String, u64, true> as ArrowBinding>::new_builder(1);
    <Map<String, u64, true> as ArrowBinding>::append_value(
        &mut builder,
        &Map::new(vec![("a".to_string(), 1)]),
    );
    let array = <Map<String, u64, true> as ArrowBinding>::finish(builder);
    assert!(keys_sorted(array.data_type()));
}

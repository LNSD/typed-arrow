//! A `#[derive(Record)]` struct with no fields.
//!
//! Regression: the generated views iterator had an unused lifetime, so this did not compile.

use typed_arrow::{
    arrow_array::{Array, StructArray},
    arrow_schema::{DataType, Fields},
    bridge::{ArrowBinding, ArrowBindingView},
    prelude::*,
};

#[derive(Record, Debug, Clone, PartialEq)]
pub struct Empty {}

#[test]
fn empty_record_is_a_struct_without_fields() {
    assert_eq!(
        <Empty as ArrowBinding>::data_type(),
        DataType::Struct(Fields::empty())
    );
}

#[test]
fn empty_record_round_trips_through_views() -> Result<(), SchemaError> {
    let mut builder = <Empty as ArrowBinding>::new_builder(3);
    for _ in 0..3 {
        <Empty as ArrowBinding>::append_value(&mut builder, &Empty {});
    }
    let array: StructArray = <Empty as ArrowBinding>::finish(builder);
    assert_eq!(array.len(), 3);
    assert_eq!(array.num_columns(), 0);

    for index in 0..array.len() {
        let view = <Empty as ArrowBindingView>::get_view(&array, index)?;
        assert_eq!(Empty::try_from(view)?, Empty {});
    }
    Ok(())
}

use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use serde_json::Value;

use crate::models::MetadataFilter;

#[derive(Parser)]
#[grammar = "filter_parser.pest"]
struct FilterParser;

/// Parse a filter expression string into a `MetadataFilter`.
/// Returns an error string suitable for returning as a 422 response.
pub fn parse_filter(input: &str) -> Result<MetadataFilter, String> {
    let pairs = FilterParser::parse(Rule::expr, input.trim())
        .map_err(|e| format!("Invalid filter expression: {e}"))?;

    let pair = pairs
        .into_iter()
        .next()
        .ok_or_else(|| "Empty filter expression".to_string())?;

    build_filter(pair)
}

fn build_filter(pair: Pair<Rule>) -> Result<MetadataFilter, String> {
    match pair.as_rule() {
        Rule::expr => {
            let inner = pair.into_inner().next().unwrap();
            build_filter(inner)
        }
        Rule::or_expr => {
            let mut children: Vec<MetadataFilter> = pair
                .into_inner()
                .map(build_filter)
                .collect::<Result<_, _>>()?;
            if children.len() == 1 {
                Ok(children.remove(0))
            } else {
                Ok(MetadataFilter { or_: Some(children), ..Default::default() })
            }
        }
        Rule::and_expr => {
            let mut children: Vec<MetadataFilter> = pair
                .into_inner()
                .map(build_filter)
                .collect::<Result<_, _>>()?;
            if children.len() == 1 {
                Ok(children.remove(0))
            } else {
                Ok(MetadataFilter { and_: Some(children), ..Default::default() })
            }
        }
        Rule::not_expr => {
            let inner = pair.into_inner().next().unwrap();
            if inner.as_rule() == Rule::atom {
                // no NOT keyword — just pass through the atom
                build_filter(inner)
            } else {
                // NOT ~ not_expr branch
                let operand = build_filter(inner)?;
                Ok(MetadataFilter { not_: Some(Box::new(operand)), ..Default::default() })
            }
        }
        Rule::atom => {
            let inner = pair.into_inner().next().unwrap();
            build_filter(inner)
        }
        Rule::comparison => {
            let inner = pair.into_inner().next().unwrap();
            build_filter(inner)
        }
        Rule::value_comparison => {
            let mut inner = pair.into_inner();
            let field = inner.next().unwrap().as_str().to_string();
            let op_pair = inner.next().unwrap();
            let op = parse_op(op_pair)?;
            let value_pair = inner.next().unwrap();
            let value = parse_value(value_pair)?;
            Ok(MetadataFilter {
                key: Some(field),
                op: Some(op),
                value: Some(value),
                ..Default::default()
            })
        }
        Rule::is_null_comparison => {
            let field = pair.into_inner().next().unwrap().as_str().to_string();
            Ok(MetadataFilter {
                key: Some(field),
                op: Some("is_null".to_string()),
                ..Default::default()
            })
        }
        Rule::is_not_null_comparison => {
            let field = pair.into_inner().next().unwrap().as_str().to_string();
            Ok(MetadataFilter {
                not_: Some(Box::new(MetadataFilter {
                    key: Some(field),
                    op: Some("is_null".to_string()),
                    ..Default::default()
                })),
                ..Default::default()
            })
        }
        rule => Err(format!("Unexpected rule: {rule:?}")),
    }
}

fn parse_op(pair: Pair<Rule>) -> Result<String, String> {
    let inner = pair.into_inner().next().unwrap();
    let op = match inner.as_rule() {
        Rule::eq  => "eq",
        Rule::neq => "neq",
        Rule::lt  => "lt",
        Rule::gt  => "gt",
        Rule::lte => "lte",
        Rule::gte => "gte",
        rule => return Err(format!("Unknown operator rule: {rule:?}")),
    };
    Ok(op.to_string())
}

fn parse_value(pair: Pair<Rule>) -> Result<Value, String> {
    let inner = pair.into_inner().next().unwrap();
    match inner.as_rule() {
        Rule::string => {
            let s = inner.as_str();
            // strip surrounding quotes and unescape \"
            let unquoted = &s[1..s.len() - 1];
            Ok(Value::String(unquoted.replace("\\\"", "\"")))
        }
        Rule::float => {
            let f: f64 = inner.as_str().parse()
                .map_err(|e| format!("Invalid float: {e}"))?;
            Ok(Value::from(f))
        }
        Rule::integer => {
            let i: i64 = inner.as_str().parse()
                .map_err(|e| format!("Invalid integer: {e}"))?;
            Ok(Value::from(i))
        }
        Rule::boolean => Ok(Value::Bool(inner.as_str().to_lowercase() == "true")),
        Rule::null => Ok(Value::Null),
        rule => Err(format!("Unknown value rule: {rule:?}")),
    }
}

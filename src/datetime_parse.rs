//! Centralized ISO 8601 datetime parsing for API inputs, metadata, and search filters.
//!
//! Accepts the formats generated clients commonly emit (RFC 3339, Python
//! `+0000` / `+00:00` offsets, chrono `Display` with a space before a colon
//! offset), matching what amgix-server / `datetime.fromisoformat` accepts.

use chrono::{DateTime, FixedOffset, NaiveDate, NaiveDateTime, Utc};

const ZONED_FMTS: &[&str] = &[
    // %z = +0000 / +00:00; %:z = +00:00 (chrono FixedOffset Display).
    "%Y-%m-%dT%H:%M:%S%.f%z",
    "%Y-%m-%dT%H:%M:%S%z",
    "%Y-%m-%dT%H:%M:%S%.f%:z",
    "%Y-%m-%dT%H:%M:%S%:z",
    // Space date/time separator (Python fromisoformat).
    "%Y-%m-%d %H:%M:%S%.f%z",
    "%Y-%m-%d %H:%M:%S%z",
    "%Y-%m-%d %H:%M:%S%.f%:z",
    "%Y-%m-%d %H:%M:%S%:z",
    // chrono Display: space before offset, e.g. "2026-08-15 13:40:24.891115219 +00:00".
    "%Y-%m-%d %H:%M:%S%.f %z",
    "%Y-%m-%d %H:%M:%S %z",
    "%Y-%m-%d %H:%M:%S%.f %:z",
    "%Y-%m-%d %H:%M:%S %:z",
    "%Y-%m-%dT%H:%M:%S%.f %z",
    "%Y-%m-%dT%H:%M:%S %z",
    "%Y-%m-%dT%H:%M:%S%.f %:z",
    "%Y-%m-%dT%H:%M:%S %:z",
];
const NAIVE_FMTS: &[&str] = &[
    "%Y-%m-%dT%H:%M:%S%.f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S%.f",
    "%Y-%m-%d %H:%M:%S",
];
const DATE: &str = "%Y-%m-%d";

fn parse_zoned_datetime(s: &str) -> Option<DateTime<FixedOffset>> {
    let s = s.trim();
    DateTime::parse_from_rfc3339(s)
        .ok()
        .or_else(|| {
            ZONED_FMTS
                .iter()
                .find_map(|fmt| DateTime::parse_from_str(s, fmt).ok())
        })
}

fn parse_naive_datetime(s: &str) -> Option<NaiveDateTime> {
    let s = s.trim();
    NAIVE_FMTS
        .iter()
        .find_map(|fmt| NaiveDateTime::parse_from_str(s, fmt).ok())
}

fn parse_date(s: &str) -> Option<NaiveDate> {
    NaiveDate::parse_from_str(s.trim(), DATE).ok()
}

/// Parse a zoned datetime and require UTC. Used for document timestamps and delete request timestamps.
pub fn parse_utc_datetime(s: &str) -> Result<DateTime<Utc>, String> {
    let s = s.trim();
    if let Some(dt) = parse_zoned_datetime(s) {
        if dt.offset().local_minus_utc() != 0 {
            return Err("Timestamp must be in UTC timezone".to_string());
        }
        return Ok(dt.with_timezone(&Utc));
    }
    if parse_naive_datetime(s).is_some() {
        return Err("Timestamp must include timezone information".to_string());
    }
    Err("Timestamp must be a valid ISO 8601 datetime string".to_string())
}

/// Whether `s` is a valid ISO 8601 datetime string (zoned, naive, or date-only).
pub fn is_valid_datetime_string(s: &str) -> bool {
    parse_zoned_datetime(s).is_some()
        || parse_naive_datetime(s).is_some()
        || parse_date(s).is_some()
}

/// Parse a datetime for search filters. Zoned values convert to UTC; naive and date-only assume UTC.
pub fn parse_datetime_as_utc(s: &str) -> Result<DateTime<Utc>, String> {
    let s = s.trim();
    if let Some(dt) = parse_zoned_datetime(s) {
        return Ok(dt.with_timezone(&Utc));
    }
    if let Some(dt) = parse_naive_datetime(s) {
        return Ok(dt.and_utc());
    }
    if let Some(d) = parse_date(s) {
        let dt = d.and_hms_opt(0, 0, 0).unwrap();
        return Ok(dt.and_utc());
    }
    Err(format!(
        "Invalid datetime value '{s}': expected ISO 8601 (e.g. '2021-05-01', '2021-05-01T00:00:00Z')"
    ))
}

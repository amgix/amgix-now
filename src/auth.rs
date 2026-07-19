//! API key authentication for the Amgix HTTP API (Qdrant-style keys and headers).

use std::collections::HashMap;

use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::AppState;

const PUBLIC_GET_PATHS: &[&str] = &["/v1/version", "/v1/health/check", "/v1/health/ready"];

const ENV_KEYS: &[(&str, KeyRole)] = &[
    ("AMGIX_SEARCH_KEY", KeyRole::Search),
    ("AMGIX_ALT_SEARCH_KEY", KeyRole::Search),
    ("AMGIX_READ_KEY", KeyRole::Read),
    ("AMGIX_ALT_READ_KEY", KeyRole::Read),
    ("AMGIX_ALT_API_KEY", KeyRole::Admin),
    ("AMGIX_API_KEY", KeyRole::Admin),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum KeyRole {
    Search = 1,
    Read = 2,
    Admin = 3,
}

#[derive(Clone, Debug)]
pub struct ApiKeyAuthConfig {
    pub need_auth: bool,
    pub key_roles: HashMap<String, KeyRole>,
}

impl ApiKeyAuthConfig {
    pub fn load_from_env() -> Self {
        let mut key_roles = HashMap::new();
        for (env_name, role) in ENV_KEYS {
            let Ok(raw) = std::env::var(env_name) else {
                continue;
            };
            let key = raw.trim();
            if key.is_empty() {
                continue;
            }
            match key_roles.get(key) {
                Some(existing) if *existing >= *role => {}
                _ => {
                    key_roles.insert(key.to_string(), *role);
                }
            }
        }
        let need_auth = !key_roles.is_empty();
        Self {
            need_auth,
            key_roles,
        }
    }

    pub fn log_startup(&self) {
        if !self.need_auth {
            tracing::info!("API key auth disabled (no AMGIX_*_KEY env vars set)");
            return;
        }
        let mut admin = 0usize;
        let mut read = 0usize;
        let mut search = 0usize;
        for role in self.key_roles.values() {
            match role {
                KeyRole::Admin => admin += 1,
                KeyRole::Read => read += 1,
                KeyRole::Search => search += 1,
            }
        }
        tracing::info!(
            "API key auth enabled ({} key(s): {} admin, {} read, {} search)",
            self.key_roles.len(),
            admin,
            read,
            search,
        );
    }
}

pub fn extract_api_key(headers: &HeaderMap) -> Option<String> {
    if let Some(value) = headers.get("api-key") {
        let Ok(raw) = value.to_str() else {
            return None;
        };
        let stripped = raw.trim();
        return if stripped.is_empty() {
            None
        } else {
            Some(stripped.to_string())
        };
    }

    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return None;
    };
    let Ok(raw) = value.to_str() else {
        return None;
    };
    if let Some(token) = raw.strip_prefix("Bearer ") {
        let stripped = token.trim();
        return if stripped.is_empty() {
            None
        } else {
            Some(stripped.to_string())
        };
    }
    let stripped = raw.trim();
    if stripped.is_empty() {
        None
    } else {
        Some(stripped.to_string())
    }
}

fn role_label(role: KeyRole) -> &'static str {
    match role {
        KeyRole::Search => "search",
        KeyRole::Read => "read",
        KeyRole::Admin => "admin",
    }
}

fn role_phrase(role: KeyRole) -> String {
    let label = role_label(role);
    let article = if label.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    };
    format!("{article} {label}")
}

fn authentication_failed_detail(needed: KeyRole) -> String {
    format!(
        "Authentication failed. This endpoint requires at least {} API key. \
         Pass it via the `api-key` header or `Authorization: Bearer <key>`.",
        role_phrase(needed),
    )
}

fn auth_error_response(detail: String) -> Response {
    let mut response = (
        StatusCode::UNAUTHORIZED,
        Json(json!({ "detail": detail })),
    )
        .into_response();
    response.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static(r#"Bearer realm="amgix", charset="UTF-8"#),
    );
    response
}

pub fn required_role(method: &str, path: &str) -> Option<KeyRole> {
    if !path.starts_with("/v1") {
        return None;
    }

    if method == "GET" && PUBLIC_GET_PATHS.contains(&path) {
        return None;
    }

    let parts: Vec<&str> = path.split('/').filter(|segment| !segment.is_empty()).collect();
    if parts.len() < 2 || parts[0] != "v1" {
        return None;
    }

    if method == "POST" && is_search_path(&parts) {
        return Some(KeyRole::Search);
    }

    if is_admin_route(method, &parts) {
        return Some(KeyRole::Admin);
    }

    Some(KeyRole::Read)
}

fn is_search_path(parts: &[&str]) -> bool {
    parts.len() >= 4 && parts[1] == "collections" && parts.last() == Some(&"search")
}

fn is_admin_route(method: &str, parts: &[&str]) -> bool {
    if parts.get(1) != Some(&"collections") {
        return false;
    }

    if method == "DELETE" {
        if parts.len() == 3 {
            return true;
        }
        if parts.iter().any(|part| *part == "documents") {
            return true;
        }
        return parts.last() == Some(&"queue");
    }

    if method != "POST" {
        return false;
    }

    if parts.len() == 3 {
        return true;
    }
    if parts.last() == Some(&"empty") || parts.last() == Some(&"bulk") {
        return true;
    }
    if parts.len() >= 5 && parts[parts.len() - 2] == "documents" && parts.last() == Some(&"sync") {
        return true;
    }
    parts.len() == 4 && parts.last() == Some(&"documents")
}

pub async fn api_auth_middleware(State(app): State<AppState>, request: Request, next: Next) -> Response {
    let auth = &app.auth;
    if !auth.need_auth {
        return next.run(request).await;
    }

    let method = request.method().as_str();
    let path = request.uri().path();
    let Some(needed) = required_role(method, path) else {
        return next.run(request).await;
    };

    let presented = extract_api_key(request.headers());
    let have = presented
        .as_ref()
        .and_then(|key| auth.key_roles.get(key).copied());
    if have.is_some_and(|role| role >= needed) {
        return next.run(request).await;
    }

    auth_error_response(authentication_failed_detail(needed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn required_role_matches_python_matrix() {
        let cases = [
            ("GET", "/v1/version", None),
            ("GET", "/v1/health/check", None),
            ("GET", "/v1/health/ready", None),
            ("GET", "/v1/collections", Some(KeyRole::Read)),
            ("POST", "/v1/collections/foo/search", Some(KeyRole::Search)),
            ("POST", "/v1/collections/foo", Some(KeyRole::Admin)),
            ("POST", "/v1/collections/foo/empty", Some(KeyRole::Admin)),
            ("POST", "/v1/collections/foo/documents", Some(KeyRole::Admin)),
            ("POST", "/v1/collections/foo/documents/sync", Some(KeyRole::Admin)),
            ("POST", "/v1/collections/foo/documents/bulk", Some(KeyRole::Admin)),
            ("POST", "/v1/collections/foo/documents/fetch", Some(KeyRole::Read)),
            ("DELETE", "/v1/collections/foo", Some(KeyRole::Admin)),
            ("DELETE", "/v1/collections/foo/documents/id", Some(KeyRole::Admin)),
            ("DELETE", "/v1/collections/foo/queue", Some(KeyRole::Admin)),
        ];
        for (method, path, expected) in cases {
            assert_eq!(required_role(method, path), expected, "{method} {path}");
        }
    }

    #[test]
    fn role_phrase_uses_an_for_admin() {
        assert_eq!(role_phrase(KeyRole::Admin), "an admin");
        assert_eq!(role_phrase(KeyRole::Read), "a read");
    }

    #[test]
    fn duplicate_keys_keep_highest_role() {
        let mut key_roles = HashMap::new();
        for (key, role) in [("shared", KeyRole::Search), ("shared", KeyRole::Read)] {
            match key_roles.get(key) {
                Some(existing) if *existing >= role => {}
                _ => {
                    key_roles.insert(key.to_string(), role);
                }
            }
        }
        assert_eq!(key_roles.get("shared"), Some(&KeyRole::Read));
    }
}

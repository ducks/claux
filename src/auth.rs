use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine as _};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{self, BufRead, BufReader, IsTerminal, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use url::Url;

const OPENROUTER_AUTH_URL: &str = "https://openrouter.ai/auth";
const OPENROUTER_EXCHANGE_URL: &str = "https://openrouter.ai/api/v1/auth/keys";
const CALLBACK_TIMEOUT: Duration = Duration::from_secs(10 * 60);

#[derive(Serialize)]
struct ExchangeRequest<'a> {
    code: &'a str,
    code_verifier: &'a str,
    code_challenge_method: &'static str,
}

#[derive(Deserialize)]
struct ExchangeResponse {
    key: String,
}

pub async fn login_openrouter(headless: bool, no_browser: bool) -> Result<()> {
    let verifier = new_code_verifier();
    let challenge = code_challenge(&verifier);

    let (authorization_url, callback) = if headless {
        (headless_authorization_url(&challenge)?, None)
    } else {
        let listener = TcpListener::bind(("127.0.0.1", 0))
            .context("could not bind the localhost OAuth callback")?;
        let callback_path = format!("/callback/{}", uuid::Uuid::new_v4().simple());
        let callback = format!(
            "http://localhost:{}{}",
            listener.local_addr()?.port(),
            callback_path
        );
        (
            callback_authorization_url(&challenge, &callback)?,
            Some((listener, callback_path)),
        )
    };

    println!("Open this URL to authorize Claux with OpenRouter:\n");
    println!("{authorization_url}\n");
    if !no_browser && !open_browser(authorization_url.as_str()) {
        eprintln!("Could not open a browser automatically; use the URL above.");
    }

    let code = match callback {
        Some((listener, callback_path)) => wait_for_callback(listener, &callback_path)?,
        None => read_authorization_code()?,
    };
    let key = exchange_code(&code, &verifier).await?;
    write_openrouter_key(&key)?;
    println!(
        "OpenRouter authentication saved to {}.",
        openrouter_credential_path()?.display()
    );
    Ok(())
}

pub fn status_openrouter() -> Result<()> {
    match read_provider_key("openrouter")? {
        Some(_) => println!(
            "OpenRouter authentication is available at {}.",
            openrouter_credential_path()?.display()
        ),
        None => println!("OpenRouter authentication is not configured."),
    }
    Ok(())
}

pub fn logout_openrouter() -> Result<()> {
    logout_provider("openrouter", "OpenRouter")
}

/// Prompt for and save an API key for a provider that does not expose an
/// OAuth flow (for example OpenCode Go or Vercel AI Gateway).
pub fn login_api_key(provider: &str, label: &str) -> Result<()> {
    let key = read_api_key(label)?;
    let path = provider_credential_path(provider)?;
    write_credential(&path, &key)?;
    println!("{label} authentication saved to {}.", path.display());
    Ok(())
}

pub fn status_provider(provider: &str, label: &str) -> Result<()> {
    match read_provider_key(provider)? {
        Some(_) => println!(
            "{label} authentication is available at {}.",
            provider_credential_path(provider)?.display()
        ),
        None => println!("{label} authentication is not configured."),
    }
    Ok(())
}

pub fn logout_provider(provider: &str, label: &str) -> Result<()> {
    let path = provider_credential_path(provider)?;
    match fs::remove_file(&path) {
        Ok(()) => println!("Removed {label} authentication from {}.", path.display()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            println!("{label} authentication was not configured.")
        }
        Err(error) => return Err(error).with_context(|| format!("remove {}", path.display())),
    }
    Ok(())
}

pub fn print_provider_token(provider: &str, label: &str) -> Result<()> {
    let key = read_provider_key(provider)?.with_context(|| {
        format!("{label} authentication is not configured; run `claux auth login {provider}`")
    })?;
    println!("{key}");
    Ok(())
}

pub fn print_openrouter_token() -> Result<()> {
    let key = read_openrouter_key()?.context(
        "OpenRouter authentication is not configured; run `claux auth login openrouter`",
    )?;
    println!("{key}");
    Ok(())
}

pub fn read_openrouter_key() -> Result<Option<String>> {
    read_provider_key("openrouter")
}

fn write_openrouter_key(key: &str) -> Result<()> {
    if key.trim().is_empty() {
        bail!("OpenRouter returned an empty API key");
    }
    write_provider_key("openrouter", key.trim())
}

fn openrouter_credential_path() -> Result<PathBuf> {
    provider_credential_path("openrouter")
}

pub fn read_provider_key(provider: &str) -> Result<Option<String>> {
    read_credential(&provider_credential_path(provider)?)
}

fn write_provider_key(provider: &str, key: &str) -> Result<()> {
    write_credential(&provider_credential_path(provider)?, key)
}

fn provider_credential_path(provider: &str) -> Result<PathBuf> {
    if provider.is_empty()
        || !provider
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
    {
        bail!("invalid provider credential name: {provider}");
    }
    let root = if let Some(path) = std::env::var_os("CLAUX_CREDENTIALS_DIR") {
        PathBuf::from(path)
    } else if cfg!(test) {
        // Unit tests must never discover credentials from the developer's home.
        std::env::temp_dir().join(format!("claux-test-credentials-{}", std::process::id()))
    } else {
        dirs::config_dir()
            .context("could not determine the user configuration directory")?
            .join("claux")
            .join("credentials")
    };
    Ok(root.join(provider))
}

fn read_api_key(label: &str) -> Result<String> {
    print!("Enter {label} API key: ");
    io::stdout().flush()?;

    let mut value = String::new();
    if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
        io::stdin().read_line(&mut value)?;
    } else {
        crossterm::terminal::enable_raw_mode()?;
        let result = (|| -> Result<()> {
            loop {
                match crossterm::event::read()? {
                    crossterm::event::Event::Key(event)
                        if event.kind == crossterm::event::KeyEventKind::Press =>
                    {
                        match event.code {
                            crossterm::event::KeyCode::Enter => break,
                            crossterm::event::KeyCode::Backspace => {
                                value.pop();
                            }
                            crossterm::event::KeyCode::Char(ch) => value.push(ch),
                            crossterm::event::KeyCode::Esc => {
                                bail!("API key entry cancelled");
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
            Ok(())
        })();
        let restore = crossterm::terminal::disable_raw_mode();
        println!();
        result?;
        restore?;
    }

    let value = value.trim().to_string();
    if value.is_empty() {
        bail!("API key was empty");
    }
    Ok(value)
}

fn write_credential(path: &Path, value: &str) -> Result<()> {
    let parent = path.parent().context("credential path has no parent")?;
    fs::create_dir_all(parent)
        .with_context(|| format!("create credential directory {}", parent.display()))?;
    set_directory_permissions(parent)?;

    let mut options = OpenOptions::new();
    options.create(true).truncate(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("open credential file {}", path.display()))?;
    set_file_permissions(path)?;
    file.write_all(value.as_bytes())?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

fn read_credential(path: &Path) -> Result<Option<String>> {
    let mut file = match fs::File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).with_context(|| format!("open {}", path.display())),
    };
    ensure_private_permissions(path)?;
    let mut value = String::new();
    file.read_to_string(&mut value)?;
    let value = value.trim().to_string();
    Ok((!value.is_empty()).then_some(value))
}

#[cfg(unix)]
fn set_directory_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_directory_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn set_file_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

#[cfg(not(unix))]
fn set_file_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn ensure_private_permissions(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;
    let mode = fs::metadata(path)?.permissions().mode();
    if mode & 0o077 != 0 {
        bail!(
            "credential file {} is accessible by other users (mode {:o}); run `chmod 600 {}`",
            path.display(),
            mode & 0o777,
            path.display()
        );
    }
    Ok(())
}

#[cfg(not(unix))]
fn ensure_private_permissions(_path: &Path) -> Result<()> {
    Ok(())
}

fn new_code_verifier() -> String {
    format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    )
}

fn code_challenge(verifier: &str) -> String {
    URL_SAFE_NO_PAD.encode(Sha256::digest(verifier.as_bytes()))
}

fn headless_authorization_url(challenge: &str) -> Result<Url> {
    let mut url = Url::parse(OPENROUTER_AUTH_URL)?;
    url.query_pairs_mut()
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256")
        .append_pair("key_label", "Claux");
    Ok(url)
}

fn callback_authorization_url(challenge: &str, callback: &str) -> Result<Url> {
    let mut url = Url::parse(OPENROUTER_AUTH_URL)?;
    url.query_pairs_mut()
        .append_pair("callback_url", callback)
        .append_pair("code_challenge", challenge)
        .append_pair("code_challenge_method", "S256");
    Ok(url)
}

fn read_authorization_code() -> Result<String> {
    print!("Paste the authorization code: ");
    std::io::stdout().flush()?;
    let mut code = String::new();
    std::io::stdin().read_line(&mut code)?;
    let code = code.trim().to_string();
    if code.is_empty() {
        bail!("authorization code was empty");
    }
    Ok(code)
}

fn wait_for_callback(listener: TcpListener, callback_path: &str) -> Result<String> {
    listener.set_nonblocking(true)?;
    println!("Waiting for the localhost callback…");
    let deadline = Instant::now() + CALLBACK_TIMEOUT;
    loop {
        match listener.accept() {
            Ok((mut stream, _)) => {
                let mut request_line = String::new();
                BufReader::new(stream.try_clone()?).read_line(&mut request_line)?;
                let result = parse_callback_request(&request_line, callback_path);
                let (status, message) = if result.is_ok() {
                    (
                        "200 OK",
                        "OpenRouter is connected. You can close this window.",
                    )
                } else {
                    (
                        "400 Bad Request",
                        "Claux could not complete authentication.",
                    )
                };
                let body = format!("<h1>{message}</h1>");
                write!(
                    stream,
                    "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                )?;
                stream.flush()?;
                return result;
            }
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    bail!("timed out waiting for OpenRouter authorization");
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => return Err(error).context("accept localhost OAuth callback"),
        }
    }
}

fn parse_callback_request(request_line: &str, callback_path: &str) -> Result<String> {
    let target = request_line
        .split_whitespace()
        .nth(1)
        .context("invalid OAuth callback request")?;
    let url = Url::parse(&format!("http://localhost{target}"))?;
    if url.path() != callback_path {
        bail!("unexpected OAuth callback path");
    }
    if let Some(error) = url
        .query_pairs()
        .find_map(|(name, value)| (name == "error").then(|| value.into_owned()))
    {
        bail!("OpenRouter authorization failed: {error}");
    }
    url.query_pairs()
        .find_map(|(name, value)| (name == "code").then(|| value.into_owned()))
        .filter(|code| !code.is_empty())
        .context("OpenRouter callback did not contain an authorization code")
}

async fn exchange_code(code: &str, verifier: &str) -> Result<String> {
    let response = reqwest::Client::new()
        .post(OPENROUTER_EXCHANGE_URL)
        .json(&ExchangeRequest {
            code,
            code_verifier: verifier,
            code_challenge_method: "S256",
        })
        .send()
        .await
        .context("exchange OpenRouter authorization code")?;
    let status = response.status();
    if !status.is_success() {
        let message = response.text().await.unwrap_or_default();
        bail!("OpenRouter token exchange failed ({status}): {message}");
    }
    let response: ExchangeResponse = response
        .json()
        .await
        .context("decode OpenRouter token exchange response")?;
    if response.key.is_empty() {
        bail!("OpenRouter token exchange returned an empty API key");
    }
    Ok(response.key)
}

fn open_browser(url: &str) -> bool {
    #[cfg(target_os = "windows")]
    let result = Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(target_os = "macos")]
    let result = Command::new("open").arg(url).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let result = Command::new("xdg-open").arg(url).spawn();
    #[cfg(not(any(unix, target_os = "windows")))]
    let result: std::io::Result<std::process::Child> = Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "unsupported platform",
    ));
    result.is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pkce_challenge_uses_sha256_base64url_without_padding() {
        assert_eq!(
            code_challenge("dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"),
            "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
        );
    }

    #[test]
    fn headless_url_has_pkce_and_label_but_no_callback() {
        let url = headless_authorization_url("challenge").unwrap();
        let query = url
            .query_pairs()
            .collect::<std::collections::HashMap<_, _>>();
        assert_eq!(query.get("code_challenge").unwrap(), "challenge");
        assert_eq!(query.get("code_challenge_method").unwrap(), "S256");
        assert_eq!(query.get("key_label").unwrap(), "Claux");
        assert!(!query.contains_key("callback_url"));
    }

    #[test]
    fn callback_parser_decodes_authorization_code() {
        assert_eq!(
            parse_callback_request(
                "GET /callback/nonce?code=abc%2F123 HTTP/1.1\r\n",
                "/callback/nonce"
            )
            .unwrap(),
            "abc/123"
        );
    }

    #[test]
    fn callback_parser_rejects_the_wrong_path() {
        assert!(parse_callback_request(
            "GET /callback/wrong?code=abc HTTP/1.1\r\n",
            "/callback/expected"
        )
        .is_err());
    }

    #[test]
    fn credential_file_round_trips_privately() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("credentials/openrouter");
        write_credential(&path, "secret").unwrap();
        assert_eq!(read_credential(&path).unwrap().as_deref(), Some("secret"));
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
    }
}

# Catalog: Secure Coding

## Detect `eval()` Usage

```yaml
# rule: no-eval.yml
id: no-eval
message: Avoid `eval()` — potential code injection.
severity: error
rule:
  any:
    - pattern: eval($$$)
    - pattern: Function($$$)
    - pattern: setTimeout($$$, $$$)
    - pattern: setInterval($$$, $$$)
  filters:
    - not:
        inside:
          kind: comment
```

## Detect `innerHTML`

```yaml
# rule: no-inner-html.yml
id: no-inner-html
message: Avoid `innerHTML` — potential XSS risk. Use `textContent` or `insertAdjacentHTML` with sanitization.
severity: warning
rule:
  any:
    - pattern: $X.innerHTML = $$$
    - pattern: $X.innerHTML += $$$
    - pattern: $X.outerHTML = $$$
```

## Detect Dangerous Shell Calls (Node.js)

```yaml
# rule: no-shell-exec.yml
id: no-shell-exec
message: Avoid shell command execution without input validation.
severity: error
rule:
  any:
    - pattern: exec($$$)
    - pattern: execSync($$$)
    - pattern: spawn($$$, { shell: true })
    - pattern: execFile($$$)
  inside:
    stopBy: end
    kind: call_expression
```

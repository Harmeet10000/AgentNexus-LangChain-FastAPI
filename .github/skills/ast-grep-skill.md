---
url: /guide/introduction.md
description: >-
  ast-grep is a tool to search and transform code. Discover its core features:
  easy syntax, flexible interface, and multi-language support.
---

# What is ast-grep?

## Introduction

ast-grep is a new AST based tool to manage your code, at massive scale.

Using ast-grep can be as simple as running a single command in your terminal:

```bash
ast-grep --pattern 'var code = $PAT' --rewrite 'let code = $PAT' --lang js
```

The command above will replace `var` statement with `let` for all JavaScript files.

***

ast-grep is a versatile tool for searching, linting and rewriting code in various languages.

* **Search**: As a *command line tool* in your terminal, `ast-grep` can precisely search code *based on AST*, running through ten thousand files in sub seconds.
* **Lint**: You can use ast-grep as a linter. Thanks to the flexible rule system, adding a new customized rule is intuitive and straightforward, with *pretty error reporting* out of box.
* **Rewrite**: ast-grep provide API to traverse and manipulate syntax tree. Besides, you can also use operators to compose complex matching from simple patterns.

> Think ast-grep as an hybrid of [grep](https://www.gnu.org/software/grep/manual/grep.html), [eslint](https://eslint.org/) and [codemod](https://github.com/facebookincubator/fastmod).

Wanna try it out? Check out the [quick start guide](/guide/quick-start)! Or see some [examples](/catalog/) to get a sense of what ast-grep can do. We also have a [playground](/playground.html) for you to try out ast-grep online!

## Supported Languages

ast-grep supports a wide range of programming languages. Here is a list of notable programming languages it supports.

|Language Domain|Supported Languages|
|:--------------|------------------:|
|System Programming| `C`, `Cpp`, `Rust`|
|Server Side Programming| `Go`, `Java`, `Python`, `C-sharp`|
|Web Development| `JS(X)`, `TS(X)`, `HTML`, `CSS`|
|Mobile App Development| `Kotlin`, `Swift`|
|Configuration | `Json`, `YAML`, `Hcl`|
|Scripting, Protocols, etc.| `Lua`, `Nix`|

Thanks to [tree-sitter](https://tree-sitter.github.io/tree-sitter/), a popular parser generator library, ast-grep manages to support [many languages](/reference/languages) out of the box!

## Motivation

Using text-based tool for searching code is fast but imprecise. We usually prefer to parse the code into [abstract syntax tree](https://www.wikiwand.com/en/Abstract_syntax_tree) for precise matches.

However, developing with AST is tedious and frustrating. Consider this "hello-world" level task: matching `console.log` in JavaScript using Babel. We will need to write code like below.

```javascript
path.parentPath.isMemberExpression() &&
path.parentPath.get('object').isIdentifier({ name: 'console' }) &&
path.parentPath.get('property').isIdentifier({ name: 'log' })
```

This snippet deserves a detailed explanation for beginners. Even for experienced developers, authoring this snippet also requires a lot of looking up references.

The pain is not language specific. The [quotation](https://portswigger.net/daily-swig/semgrep-static-code-analysis-tool-helps-eliminate-entire-classes-of-vulnerabilities) from Jobert Abma, co-founder of HackerOne, manifests the universal pain across many languages.

> The internal AST query interfaces those tools offer are often poorly documented and difficult to write, understand, and maintain.

***

ast-grep solves the problem by providing a simple core mechanism: using code to search code with the same pattern.
Consider it as same as `grep` but based on AST instead of text.

In comparison to Babel, we can complete this hello-world task in ast-grep trivially

```bash
ast-grep -p "console.log"
```

See [playground](/playground.html) in action!

Upon the simple pattern code, we can build a series of operators to compose complex matching rules for various scenarios.

Though we use JavaScript in our introduction, ast-grep is not language specific. It is a *polyglot* tool backed by the renowned library [tree-sitter](https://tree-sitter.github.io/).
The idea of ast-grep can be applied to many other languages!

## Features

There are a lot of other tools that looks like ast-grep, notable predecessors including [Semgrep](https://semgrep.dev/), [comby](https://comby.dev/), [shisho](https://github.com/flatt-security/shisho), [gogocode](https://github.com/thx/gogocode), and new comers like [gritQL](https://about.grit.io/)

What makes ast-grep stands out is:

### Performance

It is written in Rust, a native language and utilize multiple cores. (It can even beat ag when searching simple pattern). ast-grep can handle tens of thousands files in seconds.

### Progressiveness

You can start from creating a one-liner to rewrite code at command line with minimal investment. Later if you see some code smell recurrently appear in your projects, you can write a linting rule in YAML with a few patterns combined. Finally if you are a library author or framework designer, ast-grep provide programmatic interface to rewrite or transpile code efficiently.

### Pragmatism

ast-grep comes with batteries included. Interactive code modification is available. Linter and language server work out of box when you install the command line tool. ast-grep is also shipped with test framework for rule authors.

## Check out Discord and StackOverflow

Still got questions? Join our [Discord](https://discord.gg/4YZjf6htSQ) and discuss with other users!

You can also ask questions under the [ast-grep](https://stackoverflow.com/questions/tagged/ast-grep) tag on [StackOverflow](https://stackoverflow.com/questions/ask).

---

---
url: /guide/quick-start.md
description: >-
  Learn how to install ast-grep and use it to quickly find and refactor code in
  your codebase. This powerful tool can help you save time and improve the
  quality of your code.
---

# Quick Start

You can unleash `ast-grep`'s power at your fingertips in a few keystrokes on the command line!

Let's see it in action by rewriting code in a moderately large codebase: [TypeScript](https://github.com/microsoft/TypeScript/).

Our task is to rewrite old defensive code that checks nullable nested method calls to use the shiny new [optional chaining operator](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Optional_chaining) `?.`.

## Installation

First, install `ast-grep`. It is distributed by [npm](https://www.npmjs.com/package/@ast-grep/cli), [cargo](https://crates.io/crates/ast-grep), [homebrew](https://formulae.brew.sh/formula/ast-grep) and [macports](https://ports.macports.org/port/ast-grep/). You can also build it [from source](https://github.com/ast-grep/ast-grep#installation).

::: code-group

```shell [homebrew]
# install via homebrew
brew install ast-grep
```

```shell [macports]
# install via MacPorts
sudo port install ast-grep
```

```shell [nix-shell]
# try ast-grep in nix-shell
nix-shell -p ast-grep
```

```shell [cargo]
# install via cargo
cargo install ast-grep --locked
```

```shell [npm]
# install via npm
npm i @ast-grep/cli -g

# note, for pnpm, you may need manually approve postinstall script
# pnpm approve-builds
```

```shell [pip]
# install via pip
pip install ast-grep-cli
```

:::

The binary command, `ast-grep` or `sg`, should be available now. Let's try it with `--help`.

```shell
ast-grep --help
# if you are not on Linux
sg --help
```

:::danger Use `sg` on Linux
Linux has a default command `sg` for `setgroups`. You can use the full command name `ast-grep` instead of `sg`.
You can also use shorter alias if you want by `alias sg=ast-grep`.
We will use `ast-grep` in the guide below.
:::

Optionally, you can grab TypeScript source code if you want to follow the tutorial. Or you can apply the magic to your own code.

```shell
git clone git@github.com:microsoft/TypeScript.git --depth 1
```

## Pattern

Let's search for instances of calling a method on a nested property. `ast-grep` uses **patterns** to find similar code.
Think of patterns like those in our old friend `grep`, but instead of text, they match AST nodes.
We can write patterns like we write ordinary code, and it will match all code that has the same syntactical structure.

For example, the following pattern code

```javascript
obj.val && obj.val()
```

will match all the following code, regardless of white spaces or new lines.

```javascript
obj.val && obj.val() // verbatim match, of course
obj.val    &&     obj.val() // this matches, too

// this matches as well!
const result = obj.val &&
   obj.val()
```

Exact AST matching is already powerful, but we can go further with **metavariables** for more flexibility.
Use a **metavariable** to match any single AST node. Metavariables begin with `$` and are typically uppercase (e.g., `$PROP`).
Think of it like the regex dot `.`, except it matches syntax nodes, not text.

We can use the following pattern to find all property checking code.

```javascript
$PROP && $PROP()
```

This is a valid `ast-grep` pattern you can run from the command line. The `--pattern` argument specifies the target.
Optionally, use `--lang` to specify the target language.

:::code-group

```shell [Full Command]
ast-grep --pattern '$PROP && $PROP()' --lang ts TypeScript/src
```

```shell [Short Form]
ast-grep -p '$PROP && $PROP()' -l ts TypeScript/src
```

```shell [Without Lang]
# ast-grep will infer languages based on file extensions
ast-grep -p '$PROP && $PROP()' TypeScript/src
```

:::

:::tip Pro Tip
The pattern must be enclosed in single quotes `'` to prevent the shell from interpreting the `$` sign.
`ast-grep -p '$PROP && $PROP()'` is okay.

With double quotes, `ast-grep -p "$PROP && $PROP()"` would be interpreted as `ast-grep -p " && ()"` after shell expansion.
:::

## Rewrite

Cool? Now we can use this pattern to refactor the TypeScript source!

```shell
# pattern and language argument support short form
ast-grep -p '$PROP && $PROP()' \
   --rewrite '$PROP?.()' \
   --interactive \
   -l ts \
   TypeScript/src
```

`ast-grep` will start an interactive session to let you choose if you want to apply the patch.
Press `y` to accept the change!

That's it! You have refactored the TypeScript source in minutes. Congratulations!

We hope you enjoy the power of AST editing with plain programming-language patterns. Next, learn more about writing patterns.

:::tip Pattern does not work?
See our FAQ for more [guidance](/advanced/faq.html) on writing patterns.
:::

---

---
url: /guide/pattern-syntax.md
---
# Pattern Syntax

In this guide we will walk through ast-grep's pattern syntax. The example will be written in JavaScript, but the basic principle will
apply to other languages as well.

## Pattern Matching

ast-grep uses pattern code to construct AST tree and match that against target code. The pattern code can search
through the full syntax tree, so pattern can also match nested expression. For example, the pattern `a + 1` can match all the following
code.

```javascript
const b = a + 1

funcCall(a + 1)

deeplyNested({
  target: a + 1
})
```

::: warning
Pattern code must be valid code that tree-sitter can parse.

[ast-grep playground](/playground.html) is a useful tool to confirm pattern is parsed correctly.

If ast-grep fails to parse code as expected, you can try give it more context by using [object-style pattern](/reference/rule.html#pattern).
:::

## Meta Variable

It is usually desirable to write a pattern to match dynamic content.

We can use meta variables to match sub expression in pattern.

Meta variables start with the `$` sign, followed by a name composed of upper case letters `A-Z`, underscore `_` or digits `1-9`.
`$META_VARIABLE` is a wildcard expression that can match any **single** AST node.

Think it as REGEX dot `.`, except it is not textual.

:::tip Valid meta variables
`$META`, `$META_VAR`, `$META_VAR1`, `$_`, `$_123`
:::

:::danger Invalid meta variables
`$invalid`, `$Svalue`, `$123`, `$KEBAB-CASE`， `$`
:::

The pattern `console.log($GREETING)` will match all the following.

```javascript
function tryAstGrep() {
  console.log('Hello World')
}

const multiLineExpression =
  console
   .log('Also matched!')
```

But it will not match these.

```javascript
// console.log(123) in comment is not matched
'console.log(123) in string' // is not matched as well
console.log() // mismatch argument
console.log(a, b) // too many arguments
```

Note, one meta variable `$MATCH` will match one **single** AST node, so the last two `console.log` calls do not match the pattern.
Let's see how we can match multiple AST nodes.

## Multi Meta Variable

We can use `$$$` to match zero or more AST nodes, including function arguments, parameters or statements. These variables can also be named, for example: `console.log($$$ARGS)`.

### Function Arguments

For example, `console.log($$$)` can match

```javascript
console.log()                       // matches zero AST node
console.log('hello world')          // matches one node
console.log('debug: ', key, value)  // matches multiple nodes
console.log(...args)                // it also matches spread
```

### Function Parameters

`function $FUNC($$$ARGS) { $$$ }` will match

```javascript
function foo(bar) {
  return bar
}

function noop() {}

function add(a, b, c) {
  return a + b + c
}
```

:::details `ARGS` will be populated with a list of AST nodes. Click to see details.
|Code|Match|
|---|----|
|`function foo(bar) { ... }` | \[`bar`] |
|`function noop() {}` | \[] |
|`function add(a, b, c) { ... }` | \[`a`, `b`, `c`] |
:::

## Meta Variable Capturing

Meta variable is also similar to [capture group](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions/Groups_and_Backreferences) in regular expression.
You can reuse same name meta variables to find previously occurred AST nodes.

For example, the pattern `$A == $A` will have the following result.

```javascript
// will match these patterns
a == a
1 + 1 == 1 + 1
// but will not match these
a == b
1 + 1 == 2
```

### Non Capturing Match

You can also suppress meta variable capturing. All meta variables with name starting with underscore `_` will not be captured.

```javascript
// Given this pattern

$_FUNC($_FUNC)

// it will match all function call with one argument or spread call
test(a)
testFunc(1 + 1)
testFunc(...args)
```

Note in the example above, even if two meta variables have the same name `$_FUNC`, each occurrence of `$_FUNC` can match different content because they are not captured.

:::info Why use non-capturing match?
This is a useful trick to micro-optimize pattern matching speed, since we don't need to create a [HashMap](https://doc.rust-lang.org/stable/std/collections/struct.HashMap.html) for bookkeeping.
:::

### Capture Unnamed Nodes

A meta variable pattern `$META` will capture [named nodes](/advanced/core-concepts.html#named-vs-unnamed) by default.
To capture [unnamed nodes](/advanced/core-concepts.html#named-vs-unnamed), you can use double dollar sign `$$VAR`.

Namedness is an advanced topic in [Tree-sitter](https://tree-sitter.github.io/tree-sitter/using-parsers#named-vs-anonymous-nodes). You can read this [in-depth guide](/advanced/core-concepts.html) for more background.

## More Powerful Rule

Pattern is a fast and easy way to match code. But it is not as powerful as [rule](/guide/rule-config.html#rule-file) which can match code with more [precise selector](/guide/rule-config/atomic-rule.html#kind) or [more context](/guide/rule-config/relational-rule.html).

We will cover using rules in next chapter.

:::tip Pro Tip
Pattern can also be an object instead of string in YAML rule.

It is very useful to avoid ambiguity in code snippet. See [here](/guide/rule-config/atomic-rule.html#pattern) for more details.

Also see our FAQ for more [guidance](/advanced/faq.html) on writing patterns.
:::

---

---
url: /guide/rule-config.md
---
# Rule Essentials

Now you have learnt the basic of ast-grep's pattern syntax and searching.
Pattern is a handy feature for simple search. But it is not expressive enough for more complicated cases.

ast-grep provides a more sophisticated way to find your code: Rule.

Rules are like [CSS selectors](https://www.w3schools.com/cssref/css_selectors.php) that can compose together to filter AST nodes based on certain criteria.

## A Minimal Example

A minimal ast-grep rule looks like this.

```yaml
id: no-await-in-promise-all
language: TypeScript
rule:
  pattern: Promise.all($A)
  has:
    pattern: await $_
    stopBy: end
```

The *TypeScript* rule, *no-await-in-promise-all*, will find `Promise.all` that **has** `await` expression in it.

It is [suboptimal](https://github.com/hugo-vrijswijk/eslint-plugin-no-await-in-promise/) because `Promise.all` will be called [only after](https://twitter.com/hd_nvim/status/1560108625460355073) the awaited Promise resolves first.

Let's walk through the main fields in this configuration.

* `id` is a unique short string for the rule.

* `language` is the programming language that the rule is intended to check. It specifies what files will be checked against this rule, based on the file extensions. See the list of [supported languages](/reference/languages.html).

* `rule` is the most interesting part of ast-grep's configuration. It accepts a [rule object](/reference/rule.html) and defines how the rule behaves and what code will be matched. You can learn how to write rule in the [detailed guide](/guide/rule-config/atomic-rule).

## Run the Rule

There are several ways to run the rule. We will illustrate several ast-grep features here.

### `ast-grep scan --rule`

The `scan` subcommand of ast-grep CLI can run one rule at a time.

To do so, you need to save the rule above in a file on the disk, say `no-await-in-promise-all.yml`. Then you can run the following command to scan your codebase. In the example below, we are scanning a `test.ts` file.

::: code-group

```bash
ast-grep scan --rule no-await-in-promise-all.yml test.ts
```

```typescript
await Promise.all([
  await foo(),
])
```

:::

### `ast-grep scan --inline-rules`

You can also run the rule directly from the command line without saving the rule to a file. The `--inline-rules` option is useful for ad-hoc search or calling ast-grep from another program.

:::details The full inline-rules command

```bash
ast-grep scan --inline-rules '
id: no-await-in-promise-all
language: TypeScript
rule:
  pattern: Promise.all($A)
  has:
    pattern: await $_
    stopBy: end
' test.ts
```

:::

### Online Playground

ast-grep provides an online [playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IlByb21pc2UuYWxsKCRBKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJpZDogbm8tYXdhaXQtaW4tcHJvbWlzZS1hbGxcbmxhbmd1YWdlOiBUeXBlU2NyaXB0XG5ydWxlOlxuICBwYXR0ZXJuOiBQcm9taXNlLmFsbCgkQSlcbiAgaGFzOlxuICAgIHBhdHRlcm46IGF3YWl0ICRfXG4gICAgc3RvcEJ5OiBlbmQiLCJzb3VyY2UiOiJQcm9taXNlLmFsbChbXG4gIGF3YWl0IFByb21pc2UucmVzb2x2ZSgxMjMpXG5dKSJ9) to test your rule.

You can paste the rule configuration into the playground and see the matched code. The playground also has a share button that generates a link to share the rule with others.

## Rule Object

*Rule object is the core concept of ast-grep's rule system and every other features are built on top of it.*

Below is the full list of fields in a rule object. Every rule field is optional and can be omitted but at least one field should be present in a rule. A node will match a rule if and only if it satisfies all fields in the rule object.

The equivalent rule object interface in TypeScript is also provided for reference.

:::code-group

```yaml [Full Rule Object]
rule:
  # atomic rule
  pattern: 'search.pattern'
  kind: 'tree_sitter_node_kind'
  regex: 'rust|regex'
  # relational rule
  inside: { pattern: 'sub.rule' }
  has: { kind: 'sub_rule' }
  follows: { regex: 'can|use|any' }
  precedes: { kind: 'multi_keys', pattern: 'in.sub' }
  # composite rule
  all: [ {pattern: 'match.all'}, {kind: 'match_all'} ]
  any: [ {pattern: 'match.any'}, {kind: 'match_any'} ]
  not: { pattern: 'not.this' }
  matches: 'utility-rule'
```

```typescript [TS Interface]
interface RuleObject {
  // atomic rule
  pattern?: string | Pattern
  kind?: string
  regex?: string
  // relational rule
  inside?: RuleObject & Relation
  has?: RuleObject & Relation
  follows?: RuleObject & Relation
  precedes?: RuleObject & Relation
  // composite rule
  all?: RuleObject[]
  any?: RuleObject[]
  not?: RuleObject
  matches?: string
}

// See Atomic rule for explanation
interface Pattern {
  context: string
  selector: string
  strictness?: Strictness
}

// See https://ast-grep.github.io/advanced/match-algorithm.html
type Strictness =
  | 'cst'
  | 'smart'
  | 'ast'
  | 'relaxed'
  | 'signature'

// See Relation rule for explanation
interface Relation {
  stopBy?: 'neighbor' | 'end' | RuleObject
  field?: string
}
```

:::

A node must **satisfies all fields** in the rule object to be considered as a match. So the rule object can be seen as an abbreviated and **unordered** `all` rule.

:::warning Rule object is unordered!!

Unordered rule object means that certain rules may be applied before others, even if they appear later in the YAML.
Whether a node matches or not may depend on the order of rule being applied, especially when using `has`/`inside` rules.

If a rule object does not work, you can try using `all` rule to specify the order of rules. See [FAQ](/advanced/faq.html#why-is-rule-matching-order-sensitive) for more details.
:::

## Three Rule Categories

To summarize the rule object fields above, we have three categories of rules:

* **Atomic Rule**: the most basic rule that checks if AST nodes matches.
* **Relational Rule**: rules that check if a node is surrounded by another node.
* **Composite Rule**: rules that combine sub-rules together using logical operators.

These three categories of rules can be composed together to create more complex rules.

The *rule object is inspired by the CSS selectors* but with more composability and expressiveness. Think about how selectors in CSS works can help you understand the rule object!

:::tip
Don't be daunted! Learn more about how to write a rule in our [detailed guide](/guide/rule-config/atomic-rule).
:::

## Target Node

Every rule configuration will have one single root `rule`. The root rule will have *only one* AST node in one match. The matched node is called target node.
During scanning and rewriting, ast-grep will produce multiple matches to report all AST nodes that satisfies the `rule` condition as matched instances.

Though one rule match only have one AST node as matched, we can have more auxiliary nodes to display context or to perform rewrite. We will cover how rules work in details in the next page.

But for a quick primer, a rule can have a pattern and we can extract meta variables from the matched node.

For example, the rule below will match the `console.log('Hello World')`.

```yaml
rule:
  pattern: console.log($GREET)
```

And we can get `$GREET` set to `'Hello World'`.

## `language` specifies `rule` interpretation

The `language` field in the rule configuration will specify how the rule is interpreted.
For example, with `language: TypeScript`, the rule pattern `'hello world'` is parsed as TypeScript string literal.
However, the rule will have a parsing error in languages like C/Java/Rust because single quote is used for character literal and double quote should be used for string.

---

---
url: /guide/rule-config/atomic-rule.md
---
# Atomic Rule

ast-grep has three categories of rules. Let's start with the most basic one: atomic rule.

Atomic rule defines the most basic matching rule that determines whether one syntax node matches the rule or not. There are five kinds of atomic rule: `pattern`, `kind`, `regex`, `nthChild` and `range`.

## `pattern`

Pattern will match one single syntax node according to the [pattern syntax](/guide/pattern-syntax).

```yaml
rule:
  pattern: console.log($GREETING)
```

The above rule will match code like `console.log('Hello World')`.

By default, a *string* `pattern` is parsed and matched as a whole.

### Pattern Object

It is not always possible to select certain code with a simple string pattern. A pattern code can be invalid, incomplete or ambiguous for the parser since it lacks context.

For example, to select class field in JavaScript, writing `$FIELD = $INIT` will not work because it will be parsed as `assignment_expression`. See [playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiJEZJRUxEID0gJElOSVQiLCJyZXdyaXRlIjoiRGVidWcuYXNzZXJ0IiwiY29uZmlnIjoicnVsZTpcbiAgcGF0dGVybjogXG4gICAgY29udGV4dDogJ3sgJE06ICgkJCRBKSA9PiAkTUFUQ0ggfSdcbiAgICBzZWxlY3RvcjogcGFpclxuIiwic291cmNlIjoiYSA9IDEyM1xuY2xhc3MgQSB7XG4gIGEgPSAxMjNcbn0ifQ==).

***

We can also use an *object* to specify a sub-syntax node to match within a larger context. It consists of an object with three properties: `context`, `selector` and `strictness`.

* `context` (required): defines the surrounding code that helps to resolve any ambiguity in the syntax.
* `selector` (optional):  defines the sub-syntax node kind that is the actual matcher of the pattern.
* `strictness` (optional): defines how strictly pattern will match against nodes.

Let's see how pattern object can solve the ambiguity in the class field example above.

The pattern object below instructs ast-grep to select the `field_definition` node as the pattern target.

```yaml
pattern:
  selector: field_definition
  context: class A { $FIELD = $INIT }
```

ast-grep works like this:

1. First, the code in `context`, `class A { $FIELD = $INIT }`, is parsed as a class declaration.
2. Then, it looks for the `field_definition` node, specified by `selector`, in the parsed tree.
3. The selected `$FIELD = $INIT` is matched against code as the pattern.

In this way, the pattern is parsed as `field_definition` instead of  `assignment_expression`. See [playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiRGSUVMRCA9ICRJTklUIiwicmV3cml0ZSI6IkRlYnVnLmFzc2VydCIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46XG4gICAgc2VsZWN0b3I6IGZpZWxkX2RlZmluaXRpb25cbiAgICBjb250ZXh0OiBjbGFzcyBBIHsgJEZJRUxEID0gJElOSVQgfVxuIiwic291cmNlIjoiYSA9IDEyM1xuY2xhc3MgQSB7XG4gIGEgPSAxMjNcbn0ifQ==) in action.

Other examples are [function call in Go](https://github.com/ast-grep/ast-grep/issues/646) and [function parameter in Rust](https://github.com/ast-grep/ast-grep/issues/648).

### `strictness`

You can also use pattern object to control the matching strategy with `strictness` field.

By default, ast-grep uses a smart strategy to match pattern against the AST node. All nodes in the pattern must be matched, but it will skip unnamed nodes in target code.

For the definition of ***named*** and ***unnamed*** nodes, please refer to the [core concepts](/advanced/core-concepts.html) doc.

For example, the following pattern `function $A() {}` will match both plain function and async function in JavaScript. See [playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiZnVuY3Rpb24gJEEoKSB7fSIsInJld3JpdGUiOiJEZWJ1Zy5hc3NlcnQiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAneyAkTTogKCQkJEEpID0+ICRNQVRDSCB9J1xuICAgIHNlbGVjdG9yOiBwYWlyXG4iLCJzb3VyY2UiOiJmdW5jdGlvbiBhKCkge31cbmFzeW5jIGZ1bmN0aW9uIGEoKSB7fSJ9)

```js
// function $A() {}
function foo() {}    // matched
async function bar() {} // matched
```

This is because the keyword `async` is an unnamed node in the AST, so the `async` in the code to search is skipped. As long as `function`, `$A` and `{}` are matched, the pattern is considered matched.

However, this is not always the desired behavior. ast-grep provides `strictness` to control the matching strategy. At the moment, it provides these options, ordered from the most strict to the least strict:

* `cst`: All nodes in the pattern and target code must be matched. No node is skipped.
* `smart`: All nodes in the pattern must be matched, but it will skip unnamed nodes in target code. This is the default behavior.
* `ast`: Only named AST nodes in both pattern and target code are matched. All unnamed nodes are skipped.
* `relaxed`: Named AST nodes in both pattern and target code are matched. Comments and unnamed nodes are ignored.
* `signature`: Only named AST nodes' kinds are matched. Comments, unnamed nodes and text are ignored.

:::tip Deep Dive and More Examples

`strictness` is an advanced feature that you may not need in most cases.

If you are interested in more examples and details, please refer to the [deep dive](/advanced/match-algorithm.html) doc on ast-grep's match algorithm.

:::

## `kind`

Sometimes it is not easy to write a pattern because it is hard to construct the valid syntax.

For example, if we want to match class property declaration in JavaScript like `class A { a = 1 }`,
writing `a = 1` will not match the property because it is parsed as assigning to a variable.

Instead, we can use `kind` to specify the AST node type defined in [tree-sitter parser](https://tree-sitter.github.io/tree-sitter/using-parsers#named-vs-anonymous-nodes).

`kind` rule accepts the tree-sitter node's name, like `if_statement` and `expression`.
You can refer to [ast-grep playground](/playground) for relevant `kind` names.

Back to our example, we can look up class property's kind from the playground.

```yaml
rule:
  kind: field_definition
```

It will match the following code successfully ([playground link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImEgPSAxMjMiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiIyBDb25maWd1cmUgUnVsZSBpbiBZQU1MXG5ydWxlOlxuICBraW5kOiBmaWVsZF9kZWZpbml0aW9uIiwic291cmNlIjoiY2xhc3MgVGVzdCB7XG4gIGEgPSAxMjNcbn0ifQ==)).

```js
class Test {
  a = 123 // match this line
}
```

Here are some situations that you can effectively use `kind`:

1. Pattern code is ambiguous to parse, e.g. `{}` in JavaScript can be either object or code block.
2. It is too hard to enumerate all patterns of an AST kind node, e.g. matching all Java/TypeScript class declaration will need including all modifiers, generics, `extends` and `implements`.
3. Patterns only appear within specific context, e.g. the class property definition.

:::warning `kind` + `pattern` is different from pattern object
You may want to use `kind` to change how `pattern` is parsed. However, ast-grep rules are independent of each other.

To change the parsing behavior of `pattern`, you should use pattern object with `context` and `selector` field.
See [this FAQ](/advanced/faq.html#kind-and-pattern-rules-are-not-working-together-why).
:::

### ESQuery style `kind`

From ast-grep v0.39.1, you can also use ESQuery style selector in `kind` to match AST nodes.

```yaml
rule:
  kind: call_expression > identifier
```

This will match the `identifier` node that is a child of `call_expression` node. Internally, it will be converted to a [relational rule](/guide/rule-config/relational-rule.html) `has`.

See [ESQuery style kind](/reference/rule/esquery.html) for supported selectors and their equivalent rule objects.

## `regex`

The `regex` atomic rule will match the AST node by its text against a Rust regular expression.

```yaml
rule:
  regex: "\w+"
```

:::tip
The regular expression is written in [Rust syntax](https://docs.rs/regex/latest/regex/), not the popular [PCRE like syntax](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions).
So some features are not available like arbitrary look-ahead and back references.
:::

You should almost always combine `regex` with other atomic rules to make sure the regular expression is applied to the correct AST node. Regex matching is quite expensive and cannot be optimized based on AST node kinds. While `kind` and `pattern` rules can be only applied to nodes with specific `kind_id` for optimized performance.

:::tip
You can use [Rust‑style inline flags](https://docs.rs/regex/latest/regex/#grouping-and-flags), for example:

```yaml
rule:
  regex: "(?i)apple"
```

This matches Apple as well as apple or APPLE.
:::

## `nthChild`

`nthChild` is a rule to find nodes based on their indexes in the parent node's children list. In other words, it selects nodes based on their position among all sibling nodes within a parent node. It is very helpful in finding nodes without children or nodes appearing in specific positions.

`nthChild` is heavily inspired by CSS's [`nth-child` pseudo-class](https://developer.mozilla.org/en-US/docs/Web/CSS/:nth-child), and it accepts similar forms of arguments.

```yaml
# a number to match the exact nth child
nthChild: 3

# An+B style string to match position based on formula
nthChild: 2n+1

# object style nthChild rule
nthChild:
  # accepts number or An+B style string
  position: 2n+1
  # optional, count index from the end of sibling list
  reverse: true # default is false
  # optional, filter the sibling node list based on rule
  ofRule:
    kind: function_declaration # accepts ast-grep rule
```

:::tip

* `nthChild`'s index is 1-based, not 0-based, as in the CSS selector.
* `nthChild`'s node list only includes named nodes, not unnamed nodes.
  :::

**Example**

The [following rule](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiRGSUVMRCA9ICRJTklUIiwicmV3cml0ZSI6IkRlYnVnLmFzc2VydCIsImNvbmZpZyI6InJ1bGU6XG4gIGtpbmQ6IG51bWJlclxuICBudGhDaGlsZDogMiIsInNvdXJjZSI6IlsxLDIsM10ifQ==) will match the second number in the JavaScript array.

```yaml
rule:
  kind: number
  nthChild: 2
```

It will match the following code:

```js
const arr = [ 1, 2, 3, ]
            //   |- match this number
```

## `range`

`range` is a rule to match nodes based on their position in the source code. It is useful when you want to integrate external tools like compilers or type checkers with ast-grep. External tools can provide the range information of the interested node, and ast-grep can use it to rewrite the code.

`range` rule accepts a range object with `start` and `end` fields. Each field is an object with `line` and `column` fields.

```yaml
rule:
  range:
    start:
      line: 0
      column: 0
    end:
      line: 1
      column: 5
```

The above example will match an AST node having the first three characters of the first line like `foo` in `foo.bar()`.

`line` and `column` are 0-based and character-wise, and the `start` is inclusive while the `end` is exclusive.

## Tips for Writing Rules

Since one rule will have *only one* AST node in one match, it is recommended to first write the atomic rule that matches the desired node.

Suppose we want to write a rule which finds functions without a return type.
For example, this code would trigger an error:

```ts
const foo = () => {
	return 1;
}
```

The first step to compose a rule is to find the target. In this case, we can first use kind: `arrow_function` to find function node. Then we can use other rules to filter candidate nodes that does have return type.

Another trick to write cleaner rule is to use sub-rules as fields.
Please refer to [composite rule](/guide/rule-config/composite-rule.html#combine-different-rules-as-fields) for more details.

---

---
url: /guide/rule-config/relational-rule.md
---
# Relational Rules

Atomic rule can only match the target node directly. But sometimes we want to match a node based on its surrounding nodes. For example, we want to find `await` expression inside a `for` loop.

Relational rules are powerful operators that can filter the *target* nodes based on their *surrounding* nodes.

ast-grep now supports four kinds of relational rules:

`inside`, `has`, `follows`, and `precedes`.

All four relational rules accept a sub rule object as their value. The sub rule will match the surrounding node while the relational rule itself will match the target node.

## Relational Rule Example

Having an `await` expression inside a for loop is usually a bad idea because every iteration will have to wait for the previous promise to resolve.

We can use the relational rule `inside` to filter out the `await` expression.

```yaml
rule:
  pattern: await $PROMISE
  inside:
    kind: for_in_statement
    stopBy: end
```

The rule reads as "matches an `await` expression that is `inside` a `for_in_statement`".
See [Playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoiaWQ6IG5vLWF3YWl0LWluLWxvb3Bcbmxhbmd1YWdlOiBUeXBlU2NyaXB0XG5ydWxlOlxuICBwYXR0ZXJuOiBhd2FpdCAkUFJPTUlTRVxuICBpbnNpZGU6XG4gICAga2luZDogZm9yX2luX3N0YXRlbWVudFxuICAgIHN0b3BCeTogZW5kIiwic291cmNlIjoiZm9yIChsZXQgaSBvZiBbMSwgMiwzXSkge1xuICAgIGF3YWl0IFByb21pc2UucmVzb2x2ZShpKVxufSJ9).

The relational rule `inside` accepts a rule and will match any node that is inside another node that satisfies the inside rule. The `inside` rule itself matches `await` and its sub rule `kind` matches the surrounding loop.

## Relational Rule's Sub Rule

Since relational rules accept another ast-grep rule, we can compose more complex examples by using operators recursively.

```yaml
rule:
  pattern: await $PROMISE
  inside:
    any:
      - kind: for_in_statement
      - kind: for_statement
      - kind: while_statement
      - kind: do_statement
    stopBy: end
```

The above rule will match different kinds of loops, like `for`, `for-in`, `while` and `do-while`.

So all the code below matches the rule:

```js
while (foo) {
  await bar()
}
for (let i = 0; i < 10; i++) {
  await bar()
}
for (let key in obj) {
  await bar()
}
do {
  await bar()
} while (condition)
```

See in [playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoiaWQ6IG5vLWF3YWl0LWluLWxvb3Bcbmxhbmd1YWdlOiBUeXBlU2NyaXB0XG5ydWxlOlxuICBwYXR0ZXJuOiBhd2FpdCAkUFJPTUlTRVxuICBpbnNpZGU6XG4gICAgYW55OlxuICAgICAgLSBraW5kOiBmb3JfaW5fc3RhdGVtZW50XG4gICAgICAtIGtpbmQ6IGZvcl9zdGF0ZW1lbnRcbiAgICAgIC0ga2luZDogd2hpbGVfc3RhdGVtZW50XG4gICAgICAtIGtpbmQ6IGRvX3N0YXRlbWVudFxuICAgIHN0b3BCeTogZW5kIiwic291cmNlIjoid2hpbGUgKGZvbykge1xuICBhd2FpdCBiYXIoKVxufVxuZm9yIChsZXQgaSA9IDA7IGkgPCAxMDsgaSsrKSB7XG4gIGF3YWl0IGJhcigpXG59XG5mb3IgKGxldCBrZXkgaW4gb2JqKSB7XG4gIGF3YWl0IGJhcigpXG59XG5kbyB7XG4gIGF3YWl0IGJhcigpXG59IHdoaWxlIChjb25kaXRpb24pIn0=).

:::tip Pro Tip
You can also use `pattern` in relational rule! The metavariable matched in relational rule can also be used in `fix`.
This will effectively let you extract a child node from a match.
:::

## Relational Rule Mnemonics

The four relational rules can read as:

* `inside`: the *target* node must be **inside** a node that matches the sub rule.
* `has`: the *target* node must **have** a child node specified by the sub rule.
* `follows`: the *target* node must **follow** a node specified by the sub rule. (target after surrounding)
* `precedes`: the *target* node must **precede** a node specified by the sub rule. (target before surrounding).

It is sometimes confusing to remember whether the rule matches target node or surrounding node. Here is the mnemonics to help you read the rule.

First, relational rule is usually used along with another rule.

Second, the other rule will match the target node.

Finally, the relational rule's sub rule will match the surrounding node.

Together, the rule specifies that the target node will `be inside` or `follows` the surrounding node.

:::tip
All relational rule takes the form of `target` `relates` to `surrounding`.
:::

For example, the rule below will match **`hello`(target)** greeting that **follows(relation)** a **`world`(surrounding)** greeting.

```yaml
pattern: console.log('hello');
follows:
  pattern: console.log('world');
```

Consider the [input source code](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJjb25maWciOiJydWxlOlxuICBhbGw6XG4gICAgLSBwYXR0ZXJuOiBjb25zb2xlLmxvZygnaGVsbG8nKTtcbiAgICAtIGZvbGxvd3M6XG4gICAgICAgIHBhdHRlcm46IGNvbnNvbGUubG9nKCd3b3JsZCcpOyIsInNvdXJjZSI6ImNvbnNvbGUubG9nKCdoZWxsbycpOyAvLyBkb2VzIG5vdCBtYXRjaFxuY29uc29sZS5sb2coJ3dvcmxkJyk7XG5jb25zb2xlLmxvZygnaGVsbG8nKTsgLy8gbWF0Y2hlcyEhIn0=). Only the second `console.log('hello')` will match the rule.

```javascript
console.log('hello'); // does not match
console.log('world');
console.log('hello'); // matches!!
```

## Fine Tuning Relational Rule

Relational rule has several options to let you find nodes more precisely.

### `stopBy`

By default, relational rule will only match nodes one level further. For example, ast-grep will only match the direct children of the target node for the `has` rule.

You can change the behavior by using the `stopBy` field. It accepts three kinds of values: string `'end'`, string `'neighbor'` (the default option), and a rule object.

`stopBy: end` will make ast-grep search surrounding nodes until it reaches the end. For example, it stops when the rule hits root node, leaf node or the first/last sibling node.

```yaml
has:
  stopBy: end
  pattern: $MY_PATTERN
```

`stopBy` can also accept a custom rule object, so the searching will only stop when the rule matches the surrounding node.

```yaml
# find if a node is inside a function called test. It stops whenever the ancestor node is a function.
inside:
  stopBy:
    kind: function
  pattern: function test($$$) { $$$ }
```

Note the `stopBy` rule is inclusive. So when both `stopBy` rule and relational rule hit a node, the node is considered as a match.

### `field`

Sometimes it is useful to specify the node by its field. Suppose we want to find a JavaScript object property with the key `prototype`, an outdated practice that we should avoid.

```yaml
kind: pair # key-value pair in JS
has:
  field: key # note here
  regex: 'prototype'
```

This rule will match the following code

```js
var a = {
  prototype: anotherObject
}
```

but will not match this code

```js
var a = {
  normalKey: prototype
}
```

Though `pair` has a child with text `prototype` in the second example, its relative field is not `key`. That is, `prototype` is not used as `key` but instead used as value. So it does not match the rule.

---

---
url: /guide/rule-config/composite-rule.md
---
# Composite Rule

Composite rule can accept another rule or a list of rules recursively.
It provides a way to compose atomic rules into a bigger rule for more complex matching.

Below are the four composite rule operators available in ast-grep:

`all`, `any`, `not`, and `matches`.

## `all`

`all` accepts a list of rules and will match AST nodes that satisfy all the rules.

Example([playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoiaWQ6IG5vLWF3YWl0LWluLWxvb3Bcbmxhbmd1YWdlOiBUeXBlU2NyaXB0XG5ydWxlOlxuICBhbGw6XG4gICAgLSBwYXR0ZXJuOiBjb25zb2xlLmxvZygnSGVsbG8gV29ybGQnKTtcbiAgICAtIGtpbmQ6IGV4cHJlc3Npb25fc3RhdGVtZW50Iiwic291cmNlIjoiY29uc29sZS5sb2coJ0hlbGxvIFdvcmxkJyk7IC8vIG1hdGNoXG52YXIgcmV0ID0gY29uc29sZS5sb2coJ0hlbGxvIFdvcmxkJyk7IC8vIG5vIG1hdGNoIn0=)):

```yaml
rule:
  all:
    - pattern: console.log('Hello World');
    - kind: expression_statement
```

The above rule will only match a single line statement with content `console.log('Hello World');`.
But not `var ret = console.log('Hello World');` because the `console.log` call is not a statement.

We can read the rule as "matches code that is both an expression statement and has content `console.log('Hello World')`".

:::tip Pro Tip
`all` rule guarantees the order of rule matching. If you use pattern with [meta variables](/guide/pattern-syntax.html#meta-variable-capturing), make sure to use `all` array to guarantee rule execution order.
:::

## `any`

`any` accepts a list of rules and will match AST nodes as long as they satisfy any one of the rules.

Example([playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoibGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbnJ1bGU6XG4gIGFueTpcbiAgICAtIHBhdHRlcm46IHZhciBhID0gJEFcbiAgICAtIHBhdHRlcm46IGNvbnN0IGEgPSAkQVxuICAgIC0gcGF0dGVybjogbGV0IGEgPSAkQSIsInNvdXJjZSI6InZhciBhID0gMVxuY29uc3QgYSA9IDEgXG5sZXQgYSA9IDFcblxuIn0=)):

```yaml
rule:
  any:
    - pattern: var a = $A
    - pattern: const a = $A
    - pattern: let a = $A
```

The above rule will match any variable declaration statement, like `var a = 1`, `const a = 1` and `let a = 1`.

## `not`

`not` accepts a single rule and will match AST nodes that do not satisfy the rule.
Combining `not` rule and `all` can help us to filter out some unwanted matches.

Example([playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoibGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbnJ1bGU6XG4gIHBhdHRlcm46IGNvbnNvbGUubG9nKCRHUkVFVElORylcbiAgbm90OlxuICAgIHBhdHRlcm46IGNvbnNvbGUubG9nKCdIZWxsbyBXb3JsZCcpIiwic291cmNlIjoiY29uc29sZS5sb2coJ2hpJylcbmNvbnNvbGUubG9nKCdIZWxsbyBXb3JsZCcpIn0=)):

```yaml
rule:
  pattern: console.log($GREETING)
  not:
    pattern: console.log('Hello World')
```

The above rule will match any `console.log` call but not `console.log('Hello World')`.

## `matches`

`matches` is a special composite rule that takes a rule-id string. The rule-id can refer to a local utility rule defined in the same configuration file or to a global utility rule defined in the global utility rule files under separate directory. The rule will match the same nodes that the utility rule matches.

`matches` rule enable us to reuse rules and even unlock the possibility of recursive rule. It is the most powerful rule in ast-grep and deserves a separate page to explain it. Please see the [dedicated page](/guide/rule-config/utility-rule) for `matches`.

## `all` and `any` Refers to Rules, Not Nodes

`all` mean that a node should **satisfy all the rules**. `any` means that a node should **satisfy any one of the rules**.
It does not mean `all` or `any` nodes matching the rules.

For example, the rule `all: [kind: number, kind: string]` will never match any node because a node cannot be both a number and a string at the same time. New ast-grep users may think this rule should all nodes that are either a number or a string, but it is not the case.
The correct rule should be `any: [kind: number, kind: string]`.

Another example is to match a node that has both `number` child and `string` child. It is extremely easy to [write a rule](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImE6IExpc3RbJEJdIiwicmV3cml0ZSI6Imxpc3RbJEJdIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiJnZW5lcmljX3R5cGUiLCJjb25maWciOiJydWxlOlxuICBraW5kOiBhcmd1bWVudHNcbiAgaGFzOlxuICAgIGFsbDogW3traW5kOiBudW1iZXJ9LCB7IGtpbmQ6IHN0cmluZ31dIiwic291cmNlIjoibG9nKCdzdHInLCAxMjMpIn0=) like below

```yaml
has:
  all: [kind: number, kind: string]
```

It is very tempting to think that this rule will work. However, `all` rule works independently and does not rely on its containing rule `has`. Since the `all` rule matches no node, the `has` rule will also match no node.

**An ast-grep rule tests one node at a time, independently.** A rule can never test multiple nodes at once.
So the rule above means *"match a node has a child that is both a number and a string at the same time"*, which is impossible.
Instead we should search *"a node that has a number child and has a string child"*.

Here is [the correct rule](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImE6IExpc3RbJEJdIiwicmV3cml0ZSI6Imxpc3RbJEJdIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiJnZW5lcmljX3R5cGUiLCJjb25maWciOiJydWxlOlxuICBraW5kOiBhcmd1bWVudHNcbiAgYWxsOlxuICAtIGhhczogeyBraW5kOiBudW1iZXIgfVxuICAtIGhhczogeyBraW5kOiBzdHJpbmcgfSIsInNvdXJjZSI6ImxvZygnc3RyJywgMTIzKSJ9). Note `all` is used before `has`.

```yaml
all:
- has: {kind: number}
- has: {kind: string}
```

Composite rule is inspired by logical operator `and`/`or` and related list method like [`all`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.all)/[`any`](https://doc.rust-lang.org/std/iter/trait.Iterator.html#method.any). It tests whether a node matches all/any of the rules in the list.

## Combine Different Rules as Fields

Sometimes it is necessary to match node nested within other desired nodes. We can use composite rule `all` and relational `inside` to find them, but the result rule is highly nested.

For example, we want to find the usage of `this.foo` in a class getter, we can write the following rule:

```yaml
rule:
  all:
    - pattern: this.foo                              # the root node
    - inside:                                        # inside another node
        all:
          - pattern:
              context: class A { get $_() { $$$ } }  # a class getter inside
              selector: method_definition
          - inside:                                  # class body
              kind: class_body
        stopBy:                                      # but not inside nested
          any:
            - kind: object                           # either object
            - kind: class_body                       # or class
```

See the [playground link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNsYXNzIEEge1xuICAgIGdldCB0ZXN0KCkge31cbn0iLCJjb25maWciOiIjIENvbmZpZ3VyZSBSdWxlIGluIFlBTUxcbnJ1bGU6XG4gIGFsbDpcbiAgICAtIHBhdHRlcm46IHRoaXMuZm9vXG4gICAgLSBpbnNpZGU6XG4gICAgICAgIGFsbDpcbiAgICAgICAgICAtIHBhdHRlcm46XG4gICAgICAgICAgICAgIGNvbnRleHQ6IGNsYXNzIEEgeyBnZXQgJEdFVFRFUigpIHsgJCQkIH0gfVxuICAgICAgICAgICAgICBzZWxlY3RvcjogbWV0aG9kX2RlZmluaXRpb25cbiAgICAgICAgICAtIGluc2lkZTpcbiAgICAgICAgICAgICAgaW1tZWRpYXRlOiB0cnVlXG4gICAgICAgICAgICAgIGtpbmQ6IGNsYXNzX2JvZHlcbiAgICAgICAgc3RvcEJ5OlxuICAgICAgICAgIGFueTpcbiAgICAgICAgICAgIC0ga2luZDogb2JqZWN0XG4gICAgICAgICAgICAtIGtpbmQ6IGNsYXNzX2JvZHkiLCJzb3VyY2UiOiJjbGFzcyBBIHtcbiAgZ2V0IHRlc3QoKSB7XG4gICAgdGhpcy5mb29cbiAgICBsZXQgbm90VGhpcyA9IHtcbiAgICAgIGdldCB0ZXN0KCkge1xuICAgICAgICB0aGlzLmZvb1xuICAgICAgfVxuICAgIH1cbiAgfVxuICBub3RUaGlzKCkge1xuICAgIHRoaXMuZm9vXG4gIH1cbn1cbmNvbnN0IG5vdFRoaXMgPSB7XG4gIGdldCB0ZXN0KCkge1xuICAgIHRoaXMuZm9vXG4gIH1cbn0ifQ==).

To avoid such nesting-hell code (remember [callback hell](http://callbackhell.com/)?), we can use combine different rules as fields into one rule object. A rule object can have all the atomic/relational/composite rule fields because they have different names. A node will match the rule object if and only if all the rules in its fields match the node. Put in another way, they are equivalent to having an `all` rule with sub rules mentioned in fields.

For example, consider this rule.

```yaml
pattern: this.foo
inside:
  kind: class_body
```

It is equivalent to the `all` rule, regardless of the rule order.

```yaml
all:
  - pattern: this.foo
  - inside:
      kind: class_body
```

Back to our `this.foo` in getter example, we can rewrite the rule as below.

```yaml
rule:
  pattern: this.foo
  inside:
    pattern:
      context: class A { get $GETTER() { $$$ } }
      selector: method_definition
    inside:
        kind: class_body
    stopBy:
      any:
        - kind: object
        - kind: class_body
```

It has less indentation than before. See the rewritten rule [in action](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNsYXNzIEEge1xuICAgIGdldCB0ZXN0KCkge31cbn0iLCJjb25maWciOiIjIENvbmZpZ3VyZSBSdWxlIGluIFlBTUxcbnJ1bGU6XG4gIHBhdHRlcm46IHRoaXMuZm9vXG4gIGluc2lkZTpcbiAgICBwYXR0ZXJuOlxuICAgICAgY29udGV4dDogY2xhc3MgQSB7IGdldCAkR0VUVEVSKCkgeyAkJCQgfSB9XG4gICAgICBzZWxlY3RvcjogbWV0aG9kX2RlZmluaXRpb25cbiAgICBpbnNpZGU6XG4gICAgICAgIGltbWVkaWF0ZTogdHJ1ZVxuICAgICAgICBraW5kOiBjbGFzc19ib2R5XG4gICAgc3RvcEJ5OlxuICAgICAgYW55OlxuICAgICAgICAtIGtpbmQ6IG9iamVjdFxuICAgICAgICAtIGtpbmQ6IGNsYXNzX2JvZHkiLCJzb3VyY2UiOiJjbGFzcyBBIHtcbiAgZ2V0IHRlc3QoKSB7XG4gICAgdGhpcy5mb29cbiAgICBsZXQgbm90VGhpcyA9IHtcbiAgICAgIGdldCB0ZXN0KCkge1xuICAgICAgICB0aGlzLmZvb1xuICAgICAgfVxuICAgIH1cbiAgfVxuICBub3RUaGlzKCkge1xuICAgIHRoaXMuZm9vXG4gIH1cbn1cbmNvbnN0IG5vdFRoaXMgPSB7XG4gIGdldCB0ZXN0KCkge1xuICAgIHRoaXMuZm9vXG4gIH1cbn0ifQ==).

:::danger Rule object does not guarantee rule matching order
Rule object does not guarantee the order of rule matching. It is possible that the `inside` rule matches before the `pattern` rule in the example above.
:::

Rule order is not important if rules are completely independent. However, matching metavariable in patterns depends on the result of previous pattern matching. If you use pattern with [meta variables](/guide/pattern-syntax.html#meta-variable-capturing), make sure to use `all` array to guarantee rule execution order.

---

---
url: /guide/rule-config/utility-rule.md
---
# Reusing Rule as Utility

ast-grep chooses to use YAML for rule representation. While this decision makes writing rules easier, it does impose some limitations on the rule authoring.
One of the limitations is that rule objects cannot be reused. Let's see an example.

Suppose we want to match all literal values in JavaScript. We will need to match these kinds:

```yaml
any:
  - kind: 'false'
  - kind: undefined
  - kind: 'null'
  - kind: 'true'
  - kind: regex
  - kind: number
  - kind: string
```

If we want to match functions in different places using only the plain YAML file, we will have to copy and paste the rule above several times. Say, we want to match either literal values or an array of literal values:

```yaml
rule:
  any:
    - kind: 'false'
    - kind: undefined
    # more literal kinds omitted
    # ...
    - kind: array
      has:
        any:
          - kind: 'false'
          - kind: undefined
          # more literal kinds omitted
          # ...
```

ast-grep provides a mechanism to reuse common rules: `utils`. A utility rule is a rule defined in the `utils` section of the config file, or in a separate global rule file. It can be referenced in other rules using the composite rule `matches`. So, the above example can be rewritten as:

```yaml
# define util rules using utils field
utils:
  # it accepts a string-keyed dictionary of rule object
  is-literal:               # rule-id
    any:                    # actual rule object
      - kind: 'false'
      - kind: undefined
      - kind: 'null'
      - kind: 'true'
      - kind: regex
      - kind: number
      - kind: string
rule:
  any:
    - matches: is-literal # reference the util!
    - kind: array
      has:
        matches: is-literal # reference it again!
```

There are two ways to define utility rules in ast-grep: *Local utility rules* and *Global Utility Rules*. They are both used in the `matches` composite rules by their ids.

## Local Utility Rules

Local utility rules are defined in the `utils` field of the config file. `utils` is a string-keyed dictionary.

The keys of the dictionary is utility rules' identifiers, which will be later used in `matches`.
Note that local utility rule identifier cannot have the same name of another local utility rule. But a local utility rule
can have the same name of another global utility rule and override the global one.

The value of the dictionary is the rule object. You can define a local utility rule using the same syntax as the `rule` field.

**Local utility rules are only available in the config file where they are defined.**

For example, the following config file defines a local utility rule `is-literal`:

```yaml
utils:
  is-literal:
    any:
      - kind: 'false'
      - kind: undefined
      - kind: 'null'
      - kind: 'true'
      - kind: regex
      - kind: number
      - kind: string
rule:
  matches: is-literal
```

The `matches` in `rule` will run the matcher rule `is-literal` against AST nodes.

Local rules must have the same language as their rule configuration file where utilities are defined. Local rules cannot have their separate `constraints` as well.

## Global Utility Rules

Global utility rules are defined in a separate file. But they are available across all rule configurations in the project.

To create global utility rules, you first need a proper ast-grep project setup like below.

```yml
my-awesome-project   # project root
  |- rules           # rule directory
  | |- my-rule.yml
  |- utils           # utils directory
  | |- is-literal.yml
  |- sgconfig.yml    # project configuration
```

Note the `utils` directory where all global utility rules will be stored. We also need to specify which directory is utility rules so that ast-grep can pick up.

In `sgconfig.yml`:

```yml
ruleDirs:
  - rules
utilDirs:
  - utils
```

Now we can define our global utility rule in the `is-literal.yml` file. A global utility rule looks like a regular rule file, but it can only have limited fields: `id`, `language`, `rule`, `constraints` and their own local rules `utils`.

```yaml
# is-literal.yml
id: is-literal
language: TypeScript
rule:
  any:
    - kind: 'false'
    - kind: undefined
    - kind: 'null'
    - kind: 'true'
    - kind: regex
    - kind: number
    - kind: string
```

Contrary to local utility rule, you must define `id` and `language` in the global utility rule. The `id` is not defined as a dictionary key.

Global utility rule have their own local utility rules and these local rules can only be accessed in their defining global rule file. Similarly, global utility rules can have their own `constraints` as well.

Finally, a rule file, whether it is a utility rule or not, can have local utility rules with the same name of another global utility rule. Global utility rules are superseded by the local homonymous rule.

## Recursive Rule Trick

You can use a utility rule inside another utility. Besides rule reusing, this also opens the possibility of recursive rule.

For example, if we want to match all expressions that evaluate to number literal in JavaScript. We can use `kind: number` to match `123` or `1.23`. But how to match expressions in parenthesis like `(((123)))`?

Using `matches` and utility rule can solve this.

```yml
utils:
  is-number:
    any:
      - kind: number
      - kind: parenthesized_expression
        has:
          matches: is-number
rule:
  matches: is-number
```

If we matches `(123)` with this rule, we will first match the `kind: parenthesized_expression` with a direct child that also matches `is-number` rule. This will make us match `123` with `is-number` which will succeed because `kind: number` matches the number literal.

Using `matches` and recursive utility rule can unlock a lot of sophisticated usage of rule. But there is one thing you need to bear in mind:

:::danger Dependency Cycle is not allowed
Rule cannot have a cyclic dependency when using `matches`. That is, a rule cannot transitively reference itself in its composite components.
:::

A dependency cycle in rule will cause infinite recursion and make ast-grep stuck in one AST node without any progression.

However, you can use self-referencing rule in relational components like `inside` or `has`. A curious reader can try to answer why this is okay.

---

---
url: /guide/scan-project.md
---
# Scan Your Project!

Let's explore its power to run scan on your code repository in a scalable way!

`ast-grep scan` is the command you can use to run multiple rules against your repository so that you don't need to pass pattern query to your command line every time.

However, to ast-grep's scan need some scaffolding for project setup. We will walk through the process in this guide.

:::tip
`ast-grep scan` requires at least one file and one directory to work:

* `sgconfig.yml`, the [project configuration](/reference/sgconfig.html) file
* a directory storing rule files, usually `rules/`
  :::

## Create Scaffolding

To set up ast-grep's scanning, you can simply run the command `ast-grep new` in the root directory of your repository. You will be guided with a series of interactive questions, like the following:

```markdown
No sgconfig.yml found. Creating a new ast-grep project...
> Where do you want to have your rules? rules
> Do you want to create rule tests? Yes
> Where do you want to have your tests? rule-tests
> Do you want to create folder for utility rules? Yes
> Where do you want to have your utilities? utils
Your new ast-grep project has been created!
```

After you answering these questions, you will get a folder structure like the below.

```bash
my-awesome-project
  |- rules           # where rules go
  |- rule-tests       # test cases for rules
  |- utils           # global utility rules for reusing
  |- sgconfig.yml    # root configuration file
```

## Create the Rule

Now you can start creating a rule! Continue using `ast-grep new`, it will ask you what to create. But you can also use `ast-grep new rule` to create a rule directly!

You will be asked several questions about the rule going to be created. Suppose we want to create a rule to ensure no eval in JavaScript.

```markdown
> What is your rule's name? no-eval
> Choose rule's language JavaScript
Created rules at ./rules/no-eval.yml
> Do you also need to create a test for the rule? Yes
Created test at rule-tests/no-eval-test.yml
```

Now you can see open the new rule created in the `rules/no-eval.yml`. File path might vary depending on your choice on the first step.

> `no-eval.yml`

```yml
id: no-eval
message: Add your rule message here....
severity: error # error, warning, hint, info
language: JavaScript
rule:
  pattern: Your Rule Pattern here...
# utils: Extract repeated rule as local utility here.
# note: Add detailed explanation for the rule.
```

We will go through the rule config in the next chapter. But these configurations are quite obvious and self explaining.

Let's change the `pattern` inside `rule` and change the rule's message.

```yml
id: no-eval
message: Add your rule message here.... # [!code --]
message: Do not use eval! Dangerous! Hazardous! Perilous! # [!code ++]
severity: error
language: JavaScript
rule:
  pattern: Your Rule Pattern here... # [!code --]
  pattern: eval($CODE) # [!code ++]
```

Okay! The pattern syntax works just like what we have learnt before.

## Scan the Code

Now you can try scanning the code! You can create a JavaScript file containing `eval` to test it.

Run `ast-grep scan` in your project, ast-grep will give you some beautiful scan report!

```bash
error[no-eval]: Add your rule message here....
  ┌─ test.js:1:1
  │
1 │ eval('hello')
  │ ^^^^^^^^^^^^^

Error: 1 error(s) found in code.
Help: Scan succeeded and found error level diagnostics in the codebase.
```

## Summary

In this section we learnt how to set up ast-grep project, create new rules using cli tool and scan problems in the repository.

To summarize the commands we used:

* `ast-grep new` - Create a new ast-grep project
* `ast-grep new rule` - Create a new rule in a rule folder.
* `ast-grep scan` - Scan the codebase with the rules in the project.

---

---
url: /guide/project/project-config.md
---
# Project Configuration

## Root Configuration File

ast-grep supports using [YAML](https://yaml.org/) to configure its linting rules to scan your code repository.
We need a root configuration file `sgconfig.yml` to specify directories where `ast-grep` can find all rules.

In your project root, add `sgconfig.yml` with content as below.

```yaml
ruleDirs:
  - rules
```

This instructs ast-grep to use all files *recursively* inside the `rules` folder as rule files.

For example, suppose we have the following file structures.

```
my-awesome-project
  |- rules
  | |- no-var.yml
  | |- no-bit-operation.yml
  | |- my_custom_rules
  |   |- custom-rule.yml
  |   |- fancy-rule.yml
  |- sgconfig.yml
  |- not-a-rule.yml
```

All the YAML files under `rules` folder will be treated as rule files by `ast-grep`, while`not-a-rule.yml` is ignored.

**Note, the [`ast-grep scan`](/reference/cli.html#scan) command requires you have an `sgconfig.yml` in your project root.**

:::tip Pro tip
We can also use directories in `node_modules` to reuse preconfigured rules published on npm!

More broadly speaking, any git hosted projects can be imported as rule sets by using [`git submodule`](https://www.git-scm.com/book/en/v2/Git-Tools-Submodules).
:::

## Project Discovery

ast-grep will try to find the `sgconfig.yml` file in the current working directory. If it is not found, it will traverse up the directory tree until it finds one. You can also specify the path to the configuration file using the `--config` option.

```bash
ast-grep scan --config path/to/config.yml
```

:::tip Global Configuration
You can put an `sgconfig.yml` in your home directory to set global configurations for `ast-grep`. XDG configuration directory is **NOT** supported yet.
:::

Project file discovery and `--config` option are also effective in the `ast-grep run` command. So you can use configurations like [custom languages](/reference/sgconfig.html#customlanguages) and [language globs](/reference/sgconfig.html#languageglobs). Note that `run` command does not require a `sgconfig.yml` file and will stil search code without it, but `scan` command will report an error if project config is not found.

## Project Inspection

You can use the [`--inspect summary`](/reference/cli/scan.html#inspect-granularity) flag to see the project directory ast-grep is using.

```bash
ast-grep scan --inspect summary
```

It will print the project directory and the configuration file path.

```bash
sg: summary|project: isProject=true,projectDir=/path/to/project
```

Output format can be found in the [GitHub issue](https://github.com/ast-grep/ast-grep/issues/1574).

---

---
url: /guide/project/lint-rule.md
---
# Lint Rule

A lint rule is a configuration file that specifies how to find, report and fix issues in the codebase.

Lint rule in ast-grep is natural extension of the core [rule object](/guide/rule-config.html).
There are several additional fields to enable even more powerful code analysis and transformation.

## Rule Example

A typical ast-grep rule file looks like this. It reports error when using `await` inside a loop since the loop can proceed *only after* the awaited Promise resolves first. See the [eslint rule](https://eslint.org/docs/latest/rules/no-await-in-loop).

```yaml
id: no-await-in-loop
language: TypeScript
rule:
  pattern: await $_
  inside:
    any:
    - kind: for_in_statement
    - kind: while_statement

# Other linting related fields
message: Don't use await inside of loops
severity: warning
note: |
  Performing an await as part of each operation is an indication that
  the program is not taking full advantage of the parallelization benefits of async/await.
```

The *TypeScript* rule, `no-await-in-loop`, will report a warning when it finds `await` **inside** a `for-in` or `while` loop.

The linter rule file is a YAML file. It has fields identical to the [rule essentials](/guide/rule-config.html) plus some linter specific fields. `id`, `language`  and `rule` are the same as in the rule essentials.

`message`, `severity` and `note` are self-descriptive linter fields. They correspond to the similar concept `Diagnostic` in the [language server protocol](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnostic) specification.

## Basic Workflow

A full configured ast-grep rule may look like daunting and complex. But the basic workflow of ast-grep rule is simple.

1. *Find*: search the nodes in the AST that match the rewriter rules (hence the name ast-grep).
2. *Rewrite*: generate a new string based on the matched meta-variables.
3. *Patch*: optionally, replace the node text with the generated fix.

The workflow above is called [*Find and Patch*](/advanced/find-n-patch.html), which is embodied in the lint rule fields:

* **Find**
  * Find a target node based on the [`rule`](/reference/rule.html)
  * Filter the matched nodes based on [`constraints`](/guide/project/lint-rule.html#constraints)
* **Patch**
  * Rewrite the matched meta-variable based on [`transform`](/guide/project/lint-rule.html#transform)
  * Replace the matched node with [`fix`](/guide/project/lint-rule.html#fix), which can use the transformed meta-variables.

## Core Rule Fields

### `rule`

`rule` is exactly the same as the [rule object](/guide/rule-config.html) in the core ast-grep configuration.

### `constraints`

We can constrain what kind of meta variables we should match.

```yaml
rule:
  pattern: console.log($GREET)
constraints:
  GREET:
    kind: identifier
```

The above rule will constraint the [`kind`](/guide/rule-config/atomic-rule.html#kind) of matched nodes to be only `identifier`.

So `console.log(name)` will match the above rule, but `console.log('Rem')` will not because the matched variable `GREET` is string.

See [playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJjb25maWciOiIjIENvbmZpZ3VyZSBSdWxlIGluIFlBTUxcbnJ1bGU6XG4gIHBhdHRlcm46IGNvbnNvbGUubG9nKCRHUkVFVClcbmNvbnN0cmFpbnRzOlxuICBHUkVFVDpcbiAgICBraW5kOiBpZGVudGlmaWVyIiwic291cmNlIjoiY29uc29sZS5sb2coJ0hlbGxvIFdvcmxkJylcbmNvbnNvbGUubG9nKGdyZWV0aW5nKVxuIn0=) in action.

:::warning
Note, constraints only applies to the single meta variable like `$ARG`, not multiple meta variable like `$$$ARGS`.
:::

:::details `constraints` is applied after `rule` and does not work inside `not`
`constraints` is a filter to further refine the matched nodes and is applied after the `rule` is matched.
So the `constraints` field cannot be used inside `not`, for example

```yml
rule:
  pattern: console.log($GREET)
  not: { pattern: console.log($STR) }
constraints:
  STR: { kind: string}
```

The intent of the above rule is to match all `console.log` call except the one with string argument.
But it will match nothing because `console.log($STR)` is exactly the same as `console.log($GREET)` before the `constraints` is applied.
The `not` and `pattern` will conflict with each other.
:::

### `transform`

`transform` is an advanced feature that allows you to transform the matched AST nodes into another string.

It is useful when you combine `transform` and `fix` to rewrite the codebase.
For example, you may want to capitalize the matched variable name, or extract a substring from the matched node.

See the [transform](/guide/rewrite/transform.html) section in rewriting guide for more details.

### `fix`

ast-grep can perform automatic rewriting to the codebase. The `fix` field in the rule configuration specifies how to rewrite the code. We can also use meta variables specified in the `rule` in `fix`. ast-grep will replace the meta-variables with the content of actual matched AST nodes.

Example:

```yaml
rule:
  pattern: console.log($GREET)
fix: console.log('Hello ' + $GREET)
```

will rewrite `console.log('World')` to `console.log('Hello ' + 'World')`.

:::warning `fix` is textual
The `fix` field is a template string and is not parsed by tree-sitter parsers.
Meta variables in `fix` will be replaced as long as they follow the meta variable syntax.
:::

An example will be like this. The meta variable `$GREET` will be replaced both in the fix `alert($GREET)` and in the fix `nonMeta$GREET`, even though the latter cannot be parsed into valid code.

## Other Linting Fields

* `message` is a concise description when the issue is reported.
* `severity` is the issue's severity. See more in [severity](/guide/project/severity.html).
* `note` is a detailed message to elaborate the message and preferably to provide actionable fix to end users.
* `labels` is a dictionary of labels to customize error reporting's code highlighting.

### `files`/`ignores`

Rules can be applied to only certain files in a codebase with `files`. `files` supports a list of glob patterns:

```yaml
files:
- "tests/**"
- "integration_tests/test.py"
```

Similarly, you can use `ignores` to ignore applying a rule to certain files. `ignores` supports a list of glob patterns:

```yaml
ignores:
- "tests/config/**"
```

`ignores` and `files` can be used together. `ignores` will be tested before `files`. See [reference](/reference/yaml.html#ignores) for more details.

:::warning Don't add `./`

Be sure to remove `./` to the beginning of your rules. ast-grep will not recognize the paths if you add `./`.

:::

Paths in both `files` and `ignores` are relative to the project root directory, that is, `sgconfig.yml`'s directory.

## Customize Code Highlighting

ast-grep will report linting issues with highlighted code span called label. A label describes an underlined region of code associated with an issue. *By default, the matched target code and its surrounding code captured by [relational rules](/guide/rule-config/relational-rule.html)*.

ast-grep further allows you to customize the highlighting style with the configuration `labels` in the rule to provide more context to the developer. **`labels` is a dictionary of which the keys are the meta-variable name without `$` and the values ares label config objects.**

The label config object contains two fields: the required `style` and the optional `message`.

* `style` specifies the category of the label. Available choices are `primary` and `secondary`.
  * `primary` describe the primary cause of an issue.
  * `secondary` provides additional context for a diagnostic.
* `message` specifies the message to be displayed along with the label.

Note, a `label` meta-variable must have a corresponding AST node in the matched code because highlighting requires a range in the code for label. That is, the **label meta-variables must be defined in `rule` or `constraints`**. Meta-variables in `transform` cannot be used in `labels` as they are not part of the matched AST node.

***

Let's see an example. Suppose we have a [rule](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46XG4gICAgY29udGV4dDogJ2NsYXNzIEggeyAkTUVUSE9EKCkgeyAkJCQgfSB9J1xuICAgIHNlbGVjdG9yOiBtZXRob2RfZGVmaW5pdGlvblxuICBpbnNpZGU6XG4gICAgcGF0dGVybjogY2xhc3MgJENMQVNTIHsgJCQkIH1cbiAgICBzdG9wQnk6IGVuZCIsInNvdXJjZSI6ImNsYXNzIE5vdENvbXBvbmVudCB7XG4gICAgbmdPbkluaXQoKSB7fVxufSJ9) that matches method declaration in a class.

```yaml
rule:
  pattern:
    context: 'class H { $METHOD() { $$$ } }'
    selector: method_definition
  inside:
    pattern: class $CLASS { $$$ }
    stopBy: end
```

Without label customization, ast-grep will highlight the method declaration (target), and the whole class declaration, captured by relational rule. We can customize the highlighting with `labels`:

```yaml
labels:
  METHOD:
    style: primary
    message: the method name
  CLASS:
    style: secondary
    message: The class name
```

Instead of highlighting the whole method declaration and class declaration, we are just highlighting the method name and class name. The `style` field specifies the highlighting style. The `message` field specifies the message to be displayed in the editor extension. See this post for a [demo](https://x.com/hd_nvim/status/1924120276939256154) and [the example](/catalog/typescript/missing-component-decorator.html) in catalog.

:::tip VSCode Extension respects `labels`
ast-grep's LSP diagnostic reporting also respects the labels configuration. Labels with messages are displayed in the editor extension as [diagnostic related information](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#diagnosticRelatedInformation). Users can jump to the label by clicking the message in the editor.
:::

## Ignore Linting Error

It is possible to ignore a single line of code in ast-grep's scanning. A developer can suppress ast-grep's error by adding `ast-grep-ignore` comment. For example, in JavaScript:

```javascript
// ast-grep-ignore
// ast-grep-ignore: <rule-id>, <more-rule-id>
```

The first comment will suppress the following line's diagnostic. The second comment will suppress one or more specific rules.
There are more options to configure ast-grep's linting behavior, please see [severity](/guide/project/severity.html) for more deep dive.

## Test and Debug Rules

After you have written your rule, you can test it with ast-grep's builtin `test` command.
Let's see it in [next section](/guide/test-rule).

:::tip Pro Tip
You can write a standalone [rule file](/reference/rule.html) and the command `ast-grep scan -r rule.yml` to perform an [ad-hoc search](/guide/tooling-overview.html#run-one-single-query-or-one-single-rule).
:::

---

---
url: /guide/test-rule.md
---
# Test Your Rule

Though it is easy to write a simple rule to match some code in ast-grep, writing a robust and comprehensive rule to cover codebase in production is still a pretty challenging work.

To alleviate this pain, ast-grep provides a builtin tool to help you test your rule. You can provide a list of `valid` cases and `invalid` cases to test against your rule.

## Basic Concepts

Ideally, a perfect rule will approve all valid code and report issues only for all invalid code. Testing a rule should also cover two categories of code accordingly. If you are familiar with [detection theory](https://en.wikipedia.org/wiki/Detection_theory), you should recognize that testing rule will involve the four scenarios tabulated below.

|Code Validity \ Rule Report | No Report | Has Report |
|----------------------------|-----------|------------|
|         Valid              | Validated |    Noisy   |
|         Invalid            | Missing   |  Reported  |

* If ast-grep reports error for invalid code, it is a correct **reported** match.
* If ast-grep reports error for valid code, it is called **noisy** match.
* If ast-grep reports nothing for invalid code, we have a **missing** match.
* If ast-grep reports nothing for valid code, it is called **validated** match.

We will see these four case status in ast-grep's test output.

## Test Setup

Let's write a test for the rule we wrote in the [previous section](/guide/rule-config.html#rule-file).

To write a test, we first need to specify a rule test directory in `sgconfig.yml`. This directory will be used to store all test cases for rules.

Suppose we have the `sgconfig.yml` as below.

```yaml{4,5}
ruleDirs:
  - rules
# testConfigs contains a list of test directories for rules.
testConfigs:
  - testDir: rule-tests
```

The configuration file should be located at a directory that looks like this.

```bash{3,5}
my-awesome-rules/
  |- rules/
  | |- no-await-in-loop.yml        # rule file
  |- rule-tests/
  | |- no-await-in-loop-test.yml   # test file
  |- sgconfig.yml
```

`rules` folder contains all rule files, while `rule-tests` folder contains all test cases for rules.

In the example, `no-await-in-loop.yml` contains the rule configuration we wrote before.

Below are all relevant files used in this example.

::: code-group

```yaml [no-await-in-loop.yml]{1}
id: no-await-in-loop
message: Don't use await inside of loops
severity: warning
language: TypeScript
rule:
  all:
    - inside:
        any:
          - kind: for_in_statement
          - kind: while_statement
        stopBy:
          end
    - pattern: await $_
```

```yaml [no-await-in-loop-test.yml]{1}
id: no-await-in-loop
valid:
  - for (let a of b) { console.log(a) }
  # .... more valid test cases
invalid:
  - async function foo() { for (var bar of baz) await bar; }
  # .... more invalid test cases
```

```yaml [sgconfig.yml]{4,5}
ruleDirs:
  - rules
# testConfigs contains a list of test directories for rules.
testConfigs:
  - testDir: rule-tests
```

:::

We will delve into `no-await-in-loop-test.yml` in next section.

## Test Case Configuration

Test configuration file is very straightforward. It contains a list of `valid` and `invalid` cases with an `id` field to specify which rule will be tested against.

`valid` is a list of source code that we **do not** expect the rule to report any issue.
`invalid` is a list of source code that we **do** expect the rule to report some issues.

```yaml
id: no-await-in-loop
valid:
  - for (let a of b) { console.log(a) }
  # .... more valid test cases
invalid:
  - async function foo() { for (var bar of baz) await bar; }
  # .... more invalid test cases
```

After writing the test configuration file, you can run `ast-grep test` in the root folder to test your rule.
We will discuss the `skip-snapshot-tests` option later.

```bash
$ ast-grep test --skip-snapshot-tests

Running 1 tests
PASS no-await-in-loop  .........................
test result: ok. 1 passed; 0 failed;
```

ast-grep will report the passed rule and failed rule. The dots behind test case id represent passed cases.

If we swap the test case and make them failed, we will get the following output.

```bash
Running 1 tests
FAIL no-await-in-loop  ...........N............M

----------- Failure Details -----------
[Noisy] Expect no-await-in-loop to report no issue, but some issues found in:

  async function foo() { for (var bar of baz) await bar; }

[Missing] Expect rule no-await-in-loop to report issues, but none found in:

  for (let a of b) { console.log(a) }

Error: test failed. 0 passed; 1 failed;
```

The output shows that we have two failed cases. One is a **noisy** match, which means ast-grep reports error for valid code. The other is a **missing** match, which means ast-grep reports nothing for invalid code.
In the test summary, we can see the cases are marked with `N` and `M` respectively.
In failure details, we can see the detailed code snippet for each case.

Besides testing code validity, we can further test rule's output like error's message and span. This is what snapshot test will cover.

## Snapshot Test

Let's rerun `ast-grep test` without `--skip-snapshot-tests` option.
This time we will get test failure that invalid code error does not have a matching snapshot.
Previously we use the `skip-snapshot-tests` option to suppress snapshot test, which is useful when you are still working on your rule. But after the rule is polished, we can create snapshot to capture the desired output of the rule.

The `--update-all` or `-U` will generate a snapshot directory for us.

```bash
my-awesome-rules/
  |- rules/
  | |- no-await-in-loop.yml               # test file
  |- rule-tests/
  | |- no-await-in-loop-test.yml          # rule file
  | |- __snapshots__/                     # snapshots folder
  | |  |- no-await-in-loop-snapshot.yml   # generated snapshot file!
  |- sgconfig.yml
```

The generated `__snapshots__` folder will store all the error output and later test run will match against them.
After the snapshot is generated, we can run `ast-grep test` again, without any option this time, and pass all the test cases!

Furthermore, when we change the rule or update the test case, we can use interactive mode to update the snapshot.

Running this command

```bash
ast-grep test --interactive
```

ast-grep will spawn an interactive session to ask you select desired snapshot updates. Example interactive session will look like this. Note the snapshot diff is highlighted in red/green color.

```diff
[Wrong] no-await-in-loop snapshot is different from baseline.
Diff:
 labels:
 - source: await bar
   style: Primary
-  start: 2
+  start: 28
   end: 37
 - source: do { await bar; } while (baz);
   style: Secondary
For Code:
  async function foo() { do { await bar; } while (baz); }

Accept new snapshot? (Yes[y], No[n], Accept All[a], Quit[q])
```

Pressing the `y` key will accept the new snapshot and update the snapshot file.

---

---
url: /guide/project/severity.md
---
# Handle Error Reports

## Severity Levels

ast-grep supports these severity levels for rules:

* `error`: The rule will report an error and fails a scan.
* `warning`: The rule will report a warning.
* `info`: The rule will report an informational message.
* `hint`: The rule will report a hint. This is the default severity level.
* `off`: The rule will disable the rule at all.

If an `error` rule is triggered, `ast-grep scan` will exit with a non-zero status code. This is useful for CI/CD pipelines to fail the build when a rule is violated.

You can configure the severity level of a rule in the rule file:

```yaml
id: rule-id
severity: error
# ... more fields
```

## Override Severity on CLI

You can override the severity level of a rule on the command line. This is useful when you want to change the severity level of a rule for a specific scan.

```bash
ast-grep scan --error rule-id --warning other-rule-id
```

You can use multiple `--error`, `--warning`, `--info`, `--hint`, and `--off` flags to override multiple rules.

## Ignore Linting Error

It is possible to ignore a single line of code in ast-grep's scanning. A developer can suppress ast-grep's error by adding `ast-grep-ignore` above the line that triggers the issue, or on the same line.

The suppression comment has the following format, in JavaScript for example:

```javascript {1,7}
console.log('hello')  // match
// ast-grep-ignore
console.log('suppressed') // suppressed
// ast-grep-ignore: no-console
console.log('suppressed') // suppressed
// ast-grep-ignore: other-rule
console.log('world') // match

// Same line suppression
console.log('suppressed') // ast-grep-ignore
console.log('suppressed') // ast-grep-ignore: no-console
```

See the [playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiRDQUxMRVIgOj0gJmZvb3t9IiwicmV3cml0ZSI6IiIsImNvbmZpZyI6ImlkOiBuby1jb25zb2xlXG5sYW5ndWFnZTogSmF2YVNjcmlwdFxucnVsZTpcbiAgcGF0dGVybjogY29uc29sZS5sb2coJEEpIiwic291cmNlIjoiY29uc29sZS5sb2coJ2hlbGxvJykgIC8vIG1hdGNoXG4vLyBhc3QtZ3JlcC1pZ25vcmVcbmNvbnNvbGUubG9nKCdzdXBwcmVzc2VkJykgLy8gc3VwcHJlc3NlZFxuLy8gYXN0LWdyZXAtaWdub3JlOiBuby1jb25zb2xlXG5jb25zb2xlLmxvZygnc3VwcHJlc3NlZCcpIC8vIHN1cHByZXNzZWRcbi8vIGFzdC1ncmVwLWlnbm9yZTogb3RoZXItcnVsZVxuY29uc29sZS5sb2coJ3dvcmxkJykgLy8gbWF0Y2hcbiJ9) in action.

These are the rules for suppression comments:

* A comment with the content `ast-grep-ignore` will suppress the following line/the same line's diagnostic.
* The magic word `ast-grep-ignore` alone will suppress *all* kinds of diagnostics.
* `ast-grep-ignore: <rule-id>` can turn off specific rules.
* You can turn off multiple rules by providing a comma-separated list in the comment. e.g. `ast-grep-ignore: rule-1, rule-2`
* Suppression comments will suppress the next line diagnostic if and only if there is no preceding ASTs on the same line.

## File Level Suppression

You can also suppress all diagnostics in a file by adding a suppression comment at the top of the file followed by an empty line. This is useful when you want to ignore all diagnostics in a file.

For example, in JavaScript:

:::code-group

```javascript [Disable all rules]
// ast-grep-ignore

// This file will not be scanned by ast-grep
// note the empty line after the suppression comment.
debugger // this line will not be scanned
console.debug('debugging') // this line will not be scanned
```

```javascript{6} [Disable sepcific rules]
// ast-grep-ignore: no-debugger

// This file will not be scanned by ast-grep
// note the empty line after the suppression comment.
debugger // this line will not trigger error
console.debug('debugging') // this line will trigger error
```

:::

To suppress the whole file, there must be [two conditions](https://github.com/ast-grep/ast-grep/issues/1541#issuecomment-2573212686) met:

* The suppression comment is on the very first line of the file.
* AND the next line (second line in file) is empty

These conditions are designed for backward compatibility.

## Report Unused Suppressions

ast-grep can report unused suppression comments in your codebase. This is useful to keep your codebase clean and to avoid suppressing issues that are no longer relevant. An example report will look like this:

```diff
help[unused-suppression]: Unused 'ast-grep-ignore' directive.
- // ast-grep-ignore
+
```

`unused-suppression` itself behaves like a `hint` rule with auto-fix.
But it is enabled, by default, only **when all rules are enabled**.

More specifically, [these conditions](https://github.com/ast-grep/ast-grep/blob/553f5e5ac577b6d2e0904c423bb5dbd27804328b/crates/cli/src/scan.rs#L68-L73) must be met:

1. No rule is [disabled](/guide/project/severity.html#override-severity-on-cli) by the `--off` flag on the CLI. `severity: off` configured in the YAML rule file does not count.
2. The CLI [`--rule`](/reference/cli/scan.html#r-rule-rule-file) flag is not used.
3. The CLI [`--inline-rules`](/reference/cli/scan.html#inline-rules-rule-text) flag is not used.
4. The CLI [`--filter`](/reference/cli/scan.html#filter-regex) flag is not used.

:::tip Unused suppression report only happens in `ast-grep scan`
If a rule is skipped during a scan, it is possible to mistakenly report a suppression comment as unused.
So running specific rules or disabling rules will not trigger the unused suppression report.
:::

You can also override the severity level of the `unused-suppression` rule on the command line. This can change the default behavior or unused-suppression reporting.

```bash
# treat unused directive as error, useful in CI/CD
ast-grep scan --error unused-suppression
# enable report even not all rules are enabled
ast-grep --rule rule.yml scan --hint unused-suppression
```

## Disallow Suppress-All Comments

By default, `ast-grep-ignore` without a rule ID suppresses *all* diagnostics on a line. This can accidentally hide issues, especially when `ast-grep-ignore: rule-id` is mistakenly written as `ast-grep-ignore rule-id` (missing the colon).

ast-grep provides a built-in rule `no-suppress-all` to disallow such suppress-all comments. When enabled, any `ast-grep-ignore` comment that does not specify a rule ID will be flagged.

```bash
# flag suppress-all comments as warning
ast-grep scan --warning=no-suppress-all
# flag suppress-all comments as error in CI/CD
ast-grep scan --error=no-suppress-all
```

With `no-suppress-all` enabled, developers are required to always specify which rule(s) to suppress. This helps track suppression counts per rule and prevents accidental suppression of multiple rules.

## Inspect Rule Severity

Finally, ast-grep provides a CLI flag [`--inspect`](/reference/cli/scan.html#inspect-granularity) to debug what rules are enabled and their severity levels. This is useful to understand the rule configuration and to debug why a rule is not triggered.

```bash
ast-grep scan --inspect entity

```

Example standard error debugging output:

```
sg: entity|rule|no-dupe-class-members: finalSeverity=Error
sg: entity|rule|no-new-symbol: finalSeverity=Error
sg: entity|rule|no-cond-assign: finalSeverity=Warning
sg: entity|rule|no-constant-condition: finalSeverity=Warning
sg: entity|rule|no-dupe-keys: finalSeverity=Error
sg: entity|rule|no-await-in-loop: finalSeverity=Warning

```

---

---
url: /guide/rewrite-code.md
---
# Rewrite Code

One of the powers of ast-grep is that it can not only find code patterns, but also transform them into new code.

For example, you may want to rename a variable, change a function call, or add a comment. ast-grep provides two ways to do this: using the `--rewrite` flag or using the `fix` key in YAML rules.

## Using `ast-grep run -p 'pat' --rewrite`

The simplest way to rewrite code is to use the `--rewrite` flag with the `ast-grep run` command. This flag takes a string argument that specifies the new code to replace the matched pattern.
For example, if you want to change all occurrences of identifier `foo` to `bar`, you can run:

```bash
ast-grep run --pattern 'foo' --rewrite 'bar' --lang python
```

This will show you a diff of the changes that will be made. If you are using interactive mode by the `--interactive` flag, ast-grep ask you if you want to apply them.

:::tip
You can also use the `--update-all` or `-U` flag to automatically accept the changes without confirmation.
:::

## Using `fix` in YAML Rule

Another way to rewrite code is to use the `fix` option in a YAML rule file. This option allows you to specify more complex and flexible rewrite rules, such as using transformations and regular expressions.

Let's look at a simple example of using `fix` in a YAML rule ([playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmIGZvbygkWCk6XG4gICRTIiwicmV3cml0ZSI6ImxvZ2dlci5sb2coJE1BVENIKSIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IGNoYW5nZV9uYW1lXG5sYW5ndWFnZTogUHl0aG9uXG5ydWxlOlxuICBwYXR0ZXJuOiB8XG4gICAgZGVmIGZvbygkWCk6XG4gICAgICAkJCRTXG5maXg6IHwtXG4gIGRlZiBiYXooJFgpOlxuICAgICQkJFNcbi0tLVxuaWQ6IGNoYW5nZV9wYXJhbVxucnVsZTpcbiAgcGF0dGVybjogZm9vKCRYKVxuZml4OiBiYXooJFgpIiwic291cmNlIjoiZGVmIGZvbyh4KTpcbiAgICByZXR1cm4geCArIDFcblxueSA9IGZvbygyKVxucHJpbnQoeSkifQ==)).
Suppose we have a Python file named `test.py` with the following content:

```python
def foo(x):
    return x + 1

y = foo(2)
print(y)
```

We want to only change the name of the function `foo` to `baz`, but not variable/method/class. We can write a YAML rule file named `change_func.yml` with the following content:

```yaml{7-9,16}
id: change_def
language: Python
rule:
  pattern: |
    def foo($X):
      $$$S
fix: |-
  def baz($X):
    $$$S

--- # this is YAML doc separator to have multiple rules in one file

id: change_param
rule:
  pattern: foo($X)
fix: baz($X)
```

The first rule matches the definition of the function `foo`, and replaces it with `baz`. The second rule matches the calls of the function `foo`, and replaces them with `baz`. Note that we use `$X` and `$$$S` as [meta](/guide/pattern-syntax.html#meta-variable) [variables](/guide/pattern-syntax.html#multi-meta-variable), which can match any expression and any statement, respectively. We can run:

```bash
ast-grep scan -r change_func.yml test.py
```

This will show us the following diff:

```python
def foo(x): # [!code --]
def baz(x): # [!code ++]
    return x + 1

y = foo(2) # [!code --]
y = baz(2) # [!code ++]
print(y)
```

We can see that the function name and parameter name are changed as we expected.

:::tip Pro Tip
You can have multiple rules in one YAML file by using the YAML document separator `---`.
This allows you to group related rules together!
:::

## Use Meta Variable in Rewrite

As we saw in the previous example, we can use [meta variables](/guide/pattern-syntax.html#meta-variable-capturing) in both the pattern and the fix parts of a YAML rule. They are like regular expression [capture groups](https://regexone.com/lesson/capturing_groups). Meta variables are identifiers that start with `$`, and they can match any syntactic element in the code, such as expressions, statements, types, etc. When we use a meta variable in the fix part of a rule, it will be replaced by whatever it matched in the pattern part.

For example, if we have a [rule](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmIGZvbygkWCk6XG4gICRTIiwicmV3cml0ZSI6ImxvZ2dlci5sb2coJE1BVENIKSIsImNvbmZpZyI6ImlkOiBzd2FwXG5sYW5ndWFnZTogUHl0aG9uXG5ydWxlOlxuICBwYXR0ZXJuOiAkWCA9ICRZXG5maXg6ICRZID0gJFgiLCJzb3VyY2UiOiJhID0gYlxuYyA9IGQgKyBlXG5mID0gZyAqIGgifQ==) like this:

```yaml
id: swap
language: Python
rule:
  pattern: $X = $Y
fix: $Y = $X
```

This rule will swap the left-hand side and right-hand side of any assignment statement. For example, if we have a code like this:

```python
a = b
c = d + e
f = g * h
```

The rule will rewrite it as:

```python
b = a
d + e = c
g * h = f
```

[Playground link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmIGZvbygkWCk6XG4gICRTIiwicmV3cml0ZSI6ImxvZ2dlci5sb2coJE1BVENIKSIsImNvbmZpZyI6ImlkOiBzd2FwXG5sYW5ndWFnZTogUHl0aG9uXG5ydWxlOlxuICBwYXR0ZXJuOiAkWCA9ICRZXG5maXg6ICRZID0gJFgiLCJzb3VyY2UiOiJhID0gYlxuYyA9IGQgKyBlXG5mID0gZyAqIGgifQ==)

Note that this may **not** be a valid or sensible code transformation, but it illustrates how meta variables work.

:::warning Append Uppercase String to Meta Variable
It will not work if you want to append a string starting with uppercase letters to a meta variable because it will be parsed as an undefined meta variable.
:::

Suppose we want to append `Name` to the meta variable `$VAR`, the fix string `$VARName` will be parsed as `$VARN` + `ame` instead. You can instead use [replace transformation](/guide/rewrite/transform.html#rewrite-with-regex-capture-groups) to create a new variable whose content is `$VAR` plus `Name`.

:::danger Non-matched meta-variable
non-matched meta-variable will be replaced by empty string in the `fix`.
:::

### Rewrite is Indentation Sensitive

ast-grep's rewrite is indentation sensitive. That is, the indentation level of a meta-variable in the fix string is preserved in the rewritten code.

For example, if we have a rule like this:

```yaml
id: lambda-to-def
language: Python
rule:
  pattern: '$B = lambda: $R'
fix: |-
  def $B():
    return $R
```

This rule will convert a lambda function to a standard `def` function. For example, if we have a code like this:

```python
b = lambda: 123
```

The rule will rewrite it as:

```python
def b():
  return 123
```

Note that the indentation level of `return $R` is preserved as two spaces in the rewritten code, even if the replacement `123` in the original code does not have indentation at all.

`fix` string's indentation is preserved relative to their position in the source code. For example, if the `lambda` appears within `if` statement, the diff will be like:

```python
if True:
    c = lambda: 456 # [!code --]
    def c():     # [!code ++]
      return 456 # [!code ++]
```

Note that the `return 456` line has an indentation of four spaces.
This is because it has two spaces indentation as a part of the fix string, and two additional spaces because the fix string as a whole is inside the `if` statement in the original code.

## Expand the Matching Range

**ast-grep rule can only fix one target node at one time by replacing the target node text with a new string.**

Using `fix` string alone is not enough to handle complex cases where we need to delete surrounding nodes like a comma, or to change surrounding brackets. We may leave redundant text in the fixed code because we cannot delete the surrounding trivials around the matched node.

To accommodate these scenarios, ast-grep's `fix` also accepts an advanced object configuration that specifies how to fix the matched AST node: `FixConfig`. It allows you to expand the matched AST node range via two additional rules.

It has one required field `template` and two optional fields `expandStart` and `expandEnd`.

`template` is the same as the string fix. Both `expandStart` and `expandEnd` accept a [rule](/guide/rule-config.html) object to specify the expansion.

`expandStart` will expand the fixing range's start until the rule is not met, while `expandEnd` will expand the fixing range's end until the rule is not met.

### Example of deleting a key-value pair in a JavaScript object

Suppose we have a JavaScript object like this:

```JavaScript
const obj = {
  Remove: 'value1'
}
const obj2 = {
  Remove: 'value1',
  Kept: 'value2',
}
```

We want to remove the key-value pair with the key `Remove` completely. Just removing the `pair` AST node is [not enough](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IkVSUk9SIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogcGFpclxuICBoYXM6XG4gICAgZmllbGQ6IGtleVxuICAgIHJlZ2V4OiBSZW1vdmVcbmZpeDogJyciLCJzb3VyY2UiOiJjb25zdCBvYmogPSB7XG4gIFJlbW92ZTogJ3ZhbHVlMSdcbn1cbmNvbnN0IG9iajIgPSB7XG4gIFJlbW92ZTogJ3ZhbHVlMScsXG4gIEtlcHQ6ICd2YWx1ZTInLFxufVxuIn0=) in `obj2` because we also need to remove the trailing comma.

We can write [a rule in playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IkVSUk9SIiwiY29uZmlnIjoibGFuZ3VhZ2U6IGphdmFzY3JpcHRcbnJ1bGU6XG4gIGtpbmQ6IHBhaXJcbiAgaGFzOlxuICAgIGZpZWxkOiBrZXlcbiAgICByZWdleDogUmVtb3ZlXG4jIHJlbW92ZSB0aGUga2V5LXZhbHVlIHBhaXIgYW5kIGl0cyBjb21tYVxuZml4OlxuICB0ZW1wbGF0ZTogJydcbiAgZXhwYW5kRW5kOiB7IHJlZ2V4OiAnLCcgfSAjIGV4cGFuZCB0aGUgcmFuZ2UgdG8gdGhlIGNvbW1hXG4iLCJzb3VyY2UiOiJjb25zdCBvYmogPSB7XG4gIFJlbW92ZTogJ3ZhbHVlMSdcbn1cbmNvbnN0IG9iajIgPSB7XG4gIFJlbW92ZTogJ3ZhbHVlMScsXG4gIEtlcHQ6ICd2YWx1ZTInLFxufVxuIn0=) like this:

```yaml
language: javascript
rule:
  kind: pair
  has:
    field: key
    regex: Remove
# remove the key-value pair and its comma
fix:
  template: ''
  expandEnd: { regex: ',' } # expand the range to the comma
```

The idea is to remove the `pair` node and expand the fixing range to the comma. The `template` is an empty string, which means we will remove the matched node completely. The `expandEnd` rule will expand the fixing range to the comma. So the eventual matched range will be `Remove: 'value1',`, comma included.

## More Advanced Rewrite

The examples above illustrate the basic usage of rewriting code with ast-grep.

ast-grep also provides more advanced features for rewriting code, such as using [transformations](/guide/rewrite/transform.html) and [rewriter rules](/guide/rewrite/rewriter.html).

These features allow you to change the matched code to desired code, like replace string using regex, slice the string, or convert the case of the string.

We will cover these advanced features in more detail in the transform doc page.

## See More in Example Catalog

If you want to see more examples of using ast-grep to rewrite code, you can check out our [example catalog](/catalog/). There you can find various use cases and scenarios where ast-grep can help you refactor and improve your code. You can also contribute your own examples and share them with other users.

---

---
url: /guide/rewrite/transform.md
---
# `transform` Code in Rewrite

Sometimes, we may want to apply some transformations to the meta variables in the fix part of a YAML rule. For example, we may want to change the case, add or remove prefixes or suffixes. ast-grep provides a `transform` key that allows us to specify such transformations.

## Use `transform` in Rewrite

`transform` accepts a **dictionary** of which:

* the *key* is the **new variable name** to be introduced and
* the *value* is a **transformation object** that specifies which meta-variable is transformed and how.

A transformation object has a key indicating which string operation will be performed on the meta variable, and the value of that key is another object (usually with the source key). Different string operation keys expect different object values.

The following is an example illustrating the syntax of a transformation object:

```yaml
transform:
  NEW_VAR:
    replace:
      source: $VAR_NAME
      replace: regex
      by: replacement
  ANOTHER_NEW_VAR:
    substring:
      source: $NEW_VAR
      startChar: 1
      endChar: -1
```

ast-grep 0.38.3+ supports string style transformations to simplify rule writing.
The above example can be simplified to one-line style like:

```yaml
transform:
  NEW_VAR: replace($VAR_NAME, replace=regex, by=replacement)
  ANOTHER_NEW_VAR: substring($NEW_VAR, startChar=1, endChar=-1)
```

## Example of Converting Generator in Python

[Converting generator expression](https://github.com/ast-grep/ast-grep/discussions/430) to list comprehension in Python is a good example to illustrate `transform`.

More concretely, we want to achieve diffs like below:

```python
"".join(i for i in iterable) # [!code --]
"".join([i for i in iterable]) # [!code ++]
```

This rule will convert the generator inside `join` to a list.

```yaml{5-11}
id: convert_generator
rule:
  kind: generator_expression
  pattern: $GEN
transform:            # 1. the transform option
  LIST:               # 2. New variable name
    substring:        # 3. the transform operation name
      source: $GEN    # 4.1 transformation source
      startChar: 1    # 4.2 transformation argument
      endChar: -1
fix: '([$LIST])'      # 5. use the new variable in fix
```

Let's discuss the API step by step:

1. The `transform` key is used to define one or more transformations that we want to apply to the meta variables in the pattern part of the rule.
2. The `LIST` key is the new variable name that we can use in `fix` or later transformation. We can choose any name as long as it does not conflict with any existing meta variable names. **Note, the new variable name does not start with `$`.**
3. The `substring` key is the transform operation name that we want to use. This operation will extract a substring from the source string based on the given start and end characters.
4. `substring` accepts an object
   1. The `source` key specifies which meta variable we want to transform. **It should have `$` prefix.** In this case, it is `$GEN` that which matches the generator expression in the code.
   2. The `startChar` and `endChar` keys specify the indices of the start and end characters of the substring that we want to extract. In this case, we want to extract everything except the wrapping parentheses, which are the first and last characters: `(` and `)`.
5. The `fix` key specifies the new code that we want to replace the matched pattern with. We use the new variable `$LIST` in the fix part, and wrap it with `[` and `]` to make it a list comprehension.

:::tip Pro Tips
Later transformations can use the variables that were transformed before. This allows you to stack string operations and achieve complex transformations.
:::

## Supported `transformation`

We have several different transformations available now. Please check out [transformation reference](/reference/yaml/transformation.html) for more details.

* `replace`: Use a regular expression to replace the text in a meta-variable with a new text.
* `substring`: Create a new string by cutting off leading and trailing characters.
* `convert`: Change the string case of a meta-variable, such as from `camelCase` to `underscore_case`.
* `rewrite`: Apply rewriter rules to a meta-variable AST and generate a new string. It is like rewriting a sub node recursively.

## Rewrite with Regex Capture Groups

The `replace` transformation allows us to use Rust regex capture groups like `(?<NAME>.*)` to capture meta-variables and reference them in the `by` field.  For example, to replace `debug` with `release` in a function name, we can use the following transformation:

```yaml
id: debug-to-release
language: js
rule: {pattern: $OLD_FN($$$ARGS)}   # Capture OLD_FN
constraints: {OLD_FN: {regex: ^debug}}  # Only match if it starts with 'debug'
transform:
  NEW_FN:
    replace:
      source: $OLD_FN
      replace: debug(?<REG>.*)      # Capture everything following 'debug' as REG
      by: release$REG               # Refer to REG just like a meta-variable
fix: $NEW_FN($$$ARGS)
```

which will result in [the following change](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IkVSUk9SIiwiY29uZmlnIjoiaWQ6IGRlYnVnLXRvLXJlbGVhc2Vcbmxhbmd1YWdlOiBqc1xucnVsZToge3BhdHRlcm46ICRPTERfRk4oJCQkQVJHUyl9ICAgIyBDYXB0dXJlIE9MRF9GTlxuY29uc3RyYWludHM6IHtPTERfRk46IHtyZWdleDogXmRlYnVnfX0gICMgT25seSBtYXRjaCBpZiBpdCBzdGFydHMgd2l0aCAnZGVidWcnXG50cmFuc2Zvcm06XG4gIE5FV19GTjpcbiAgICByZXBsYWNlOlxuICAgICAgc291cmNlOiAkT0xEX0ZOXG4gICAgICByZXBsYWNlOiBkZWJ1Zyg/PFJFRz4uKikgICAgICAjIENhcHR1cmUgZXZlcnl0aGluZyBmb2xsb3dpbmcgJ2RlYnVnJyBhcyBSRUdcbiAgICAgIGJ5OiByZWxlYXNlJFJFRyAgICAgICAgICAgICAgICMgUmVmZXIgdG8gUkVHIGp1c3QgbGlrZSBhIG1ldGEtdmFyaWFibGVcbmZpeDogJE5FV19GTigkJCRBUkdTKSIsInNvdXJjZSI6ImRlYnVnRm9vKGFyZzEsIGFyZzIpICAifQ==):

```js
debugFoo(arg1, arg2)  // [!code --]
releaseFoo(arg1, arg2)  // [!code ++]
```

Alternatively, replacing `fooDebug` with `fooRelease`, is difficult because you can't concatenate a meta-variable with a capitalized string literal. `release$REG` is fine, but `$REGRelease` will be interpreted as a single meta-variable and not a concatenation. One workaround is to use multiple sequential transformations, as shown below.

:::warning Limitation
You can only extract regex capture groups in the `replace` field of the `replace` transformation and you can only reference them in the `by` field of the same transformation. The regular `regex` rule does not support capture groups.
:::

## Multiple Sequential Transformations

Each transformation outputs a meta-variable that can be used as the input to later transformations. Chaining transformations like this allows us to build up complex behaviors.

Here we can see an example that transforms `fooDebug` into `fooRelease` by using `convert`, `replace`, and `convert` transformations.

```yaml
rule: {pattern: $OLD_FN($$$ARGS)}      # Capture OLD_FN
constraints: {OLD_FN: {regex: Debug$}} # Only match if it ends with 'Debug'
transform:
  KEBABED:                             # 1. Convert to 'foo-debug'
    convert:
      source: $OLD_FN
      toCase: kebabCase
  RELEASED:                            # 2. Replace with 'foo-release'
    replace:
      source: $KEBABED
      replace: (?<ROOT>)-debug
      by: $ROOT-release
  UNKEBABED:                           # 3. Convert to 'fooRelease'
    convert:
      source: $RELEASED
      toCase: camelCase
fix: $UNKEBABED($$$ARGS)
```

## Add conditional text

Occasionally we may want to add extra text, such as punctuations and newlines, to our fixer string. But whether we should add the new text depends on the presence of absence of other syntax nodes.

A typical scenario is adding a comma between two arguments or list items. We only want to add a comma when the item we are adding is not the last one in the argument list.

We can use `replace` transformation to create a new meta-variable that only contains text when another meta-variable matches something.

For example, suppose we want to add a new argument to existing function call. We need to add a comma `,` after the new argument only when the existing call already has some arguments.

```yaml
id: add-leading-argument
language: python
rule:
  pattern: $FUNC($$$ARGS)
transform:
  MAYBE_COMMA:
    replace:
      source: $$$ARGS
      replace: '^.+'
      by: ', '
fix:
  $FUNC(new_argument$MAYBE_COMMA$$$ARGS)
```

In the above example, if `$$$ARGS` matches nothing, it will be an empty string and the `replace` transformation will take no effect. The final fix string will be instantiated to `$FUNC(new_argument)`.

If `$$$ARGS` does match nodes, then the replacement regular expression will replace the text with `,`, so the final fix string will be
`$FUNC(new_argument, $$$ARGS)`

:::tip DasSurma Trick
This method is invented by [Surma](https://surma.dev/) in a [tweet](https://twitter.com/DasSurma/status/1706086320051794217), so the useful trick is named after him.
:::

## String Style Transformations

To simplify the syntax of transformations, ast-grep 0.38.3+ supports a new string style transformation syntax. This allows us to write transformations in a more concise and readable way.

The string style transformation syntax is similar to the CSS function call syntax

```yaml
# illustration of string style transformation syntax
NEW_VAR: transform($SOURCE_VAR, option1=value1, option2=value2)
```

The transformation name is followed by parentheses containing the arguments. The first argument is always the source meta-variable, and the rest are the transformation options in the form of key-value pairs.

For example, the transformation examples above can be written as:

```yaml
transform:
  LIST: substring($GEN, startChar=1, endChar=-1)
  KEBABED: convert($OLD_FN, toCase=kebabCase)
  MAYBE_COMMA: replace($$$ARGS, replace='^.+', by=', ')
```

:::warning
The string style transformation syntax is only available in ast-grep 0.38.3 and later versions. If you are using an older version, please use the original object style syntax.
:::

## Even More Advanced Transformations

We can use rewriters in the [`rewrite`](/guide/rewrite/rewriter.html) transformation to apply dynamic transformations to the AST. We will cover it in next section.

---

---
url: /guide/rewrite/rewriter.md
---
# Rewriter in Fix

`rewriters` allow you to apply rules to specific parts of the matching AST nodes.

ast-grep's `fix` will only replace the matched nodes, one node at a time.
But it is common to replace multiple nodes with different fixes at once. The `rewriters` field allows you to do this.

The basic workflow of `rewriters` is as follows:

1. Find a list of sub-nodes under a meta-variable that match different rewriters.
2. Generate a distinct fix for each sub-node based on the matched rewriter sub-rule.
3. Join the fixes together and store the string in a new metavariable for later use.

## Key Steps to Use Rewriters

To use rewriters, you have three steps.

**1. Define `rewriters` field in the Yaml rule root.**

```yaml
id: rewriter-demo
language: Python
rewriters:
- id: sub-rule
  rule: # some rule
  fix: # some fix
```

**2. Apply the defined rewriters to a metavariable via `transform`.**

```yaml
transform:
  NEW_VAR:
    rewrite:
      rewriters: [sub-rule]
      source: $OLD_VAR
```

**3. Use other ast-grep fields to wire them together.**

```yaml
rule: { pattern: a = $OLD_VAR }
# ... rewriters and transform
fix: a = $NEW_VAR
```

## Rewriter Example

Let's see a contrived example: converting `dict` function call to dictionary literal in Python.

### General Idea

In Python, you can create a dictionary using the `dict` function or the `{}` literal.

```python
# dict function call
d = dict(a=1, b=2)
# dictionary literal
d = {'a': 1, 'b': 2}
```

We will use the `rewriters` field to convert the `dict` function call to a dictionary literal.

The recipe is to first find the `dict` function call. Then, extract the keyword arguments like `a=1` and transform them into a dictionary key-value pair `'a': 1`. Finally, we will replace the `dict` function call by combining these transformed pairs and wrapping them in a bracket.

The key step is extraction and transformation, which is done by the `rewriters` field.

### Define a Rewriter

Our goal is to find keyword arguments in the `dict` function call and transform them into dictionary key-value pairs.

So let's first define a rule to match the keyword arguments in the `dict` function call.

```yaml
rule:
  kind: keyword_argument
  all:
  - has:
      field: name
      pattern: $KEY
  - has:
      field: value
      pattern: $VAL
```

This rule can match the keyword arguments in the `dict` function call and extract key and value in the argument to meta-variables `$KEY` and `$VAL` respectively. [For example](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoic3RhcnRfdGFnIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDoga2V5d29yZF9hcmd1bWVudFxuICBhbGw6XG4gIC0gaGFzOlxuICAgICAgZmllbGQ6IG5hbWVcbiAgICAgIHBhdHRlcm46ICRLRVlcbiAgLSBoYXM6XG4gICAgICBmaWVsZDogdmFsdWVcbiAgICAgIHBhdHRlcm46ICRWQUwiLCJzb3VyY2UiOiJkID0gZGljdChhPTEsIGI9MikifQ==), `dict(a=1)` will extract `a` to `$KEY` and `1` to `$VAL`.

Then, we define the rule as a rewriter and add fix field to transform the keyword argument to a dictionary key-value pair.

```yaml
rewriters:
- id: dict-rewrite
  rule:
    kind: keyword_argument
    all:
    - has:
        field: name
        pattern: $KEY
    - has:
        field: value
        pattern: $VAL
  fix: "'$KEY': $VAL"
```

You can see the `rewriters` field accepts a list of regular ast-grep rules. Rewriter rule must have an `id` field to identify the rewriter, a rule to specify the node to match, and a `fix` field to transform the matched node.

Applying the rule above alone will transform `a=1` to `'a': 1`. But it is not enough to replace the `dict` function call. We need to combine these pairs and wrap them in a bracket. We need to apply this rewriter to all keyword arguments and join them.

### Apply Rewriter

Now, we apply the rewriter to the `dict` function call. This is done by the `transform` field.

First, we match the `dict` function call with the pattern `dict($$$ARGS)`. The `$$$ARGS` is a special metavariable that matches all arguments of the function call. Then, we apply the rewriter `dict-rewrite` to the `$$$ARGS` and store the result in a new metavariable `LITERAL`.

```yaml
rule:
  pattern: dict($$$ARGS)        # match dict function call, capture $$$ARGS
transform:
  LITERAL:                      # the transformed code
    rewrite:
      rewriters: [dict-rewrite] # specify the rewriter defined above
      source: $$$ARGS           # apply rewriters to $$$ARGS arguments
```

ast-grep will first try match the `dict-rewrite` rule to each sub node inside `$$$ARGS`. If the node has a matching rule, ast-grep will extract the node specified by the meta-variables in the `dict-rewrite` rewriter rule. It will then generate a new string using the `fix`.
Finally, the generated strings replace the matched sub-nodes in the `$$$ARGS` and the new code is stored in the `LITERAL` metavariable.

For example, `dict(a=1, b=2)` will match the `$$$ARGS` as `a=1, b=2`. The rewriter will transform `a=1` to `'a': 1` and `b=2` to `'b': 2`. The final value of `LITERAL` will be `'a': 1, 'b': 2`.

### Combine and Replace

Finally, we combine the transformed keyword arguments and replace the `dict` function call.

```yaml
# define rewriters
rewriters:
- id: dict-rewrite
  rule:
    kind: keyword_argument
    all:
    - has:
        field: name
        pattern: $KEY
    - has:
        field: value
        pattern: $VAL
  fix: "'$KEY': $VAL"
# find the target node
rule:
  pattern: dict($$$ARGS)
# apply rewriters to sub node
transform:
  LITERAL:
    rewrite:
      rewriters: [dict-rewrite]
      source: $$$ARGS
# combine and replace
fix: '{ $LITERAL }'
```

See the final result in [action](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGljdCgkJCRBUkdTKSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiMgZGVmaW5lIHJld3JpdGVyc1xucmV3cml0ZXJzOlxuLSBpZDogZGljdC1yZXdyaXRlXG4gIHJ1bGU6XG4gICAga2luZDoga2V5d29yZF9hcmd1bWVudFxuICAgIGFsbDpcbiAgICAtIGhhczpcbiAgICAgICAgZmllbGQ6IG5hbWVcbiAgICAgICAgcGF0dGVybjogJEtFWVxuICAgIC0gaGFzOlxuICAgICAgICBmaWVsZDogdmFsdWVcbiAgICAgICAgcGF0dGVybjogJFZBTFxuICBmaXg6IFwiJyRLRVknOiAkVkFMXCJcbiMgZmluZCB0aGUgdGFyZ2V0IG5vZGVcbnJ1bGU6XG4gIHBhdHRlcm46IGRpY3QoJCQkQVJHUylcbiMgYXBwbHkgcmV3cml0ZXJzIHRvIHN1YiBub2RlXG50cmFuc2Zvcm06XG4gIExJVEVSQUw6XG4gICAgcmV3cml0ZTpcbiAgICAgIHJld3JpdGVyczogW2RpY3QtcmV3cml0ZV1cbiAgICAgIHNvdXJjZTogJCQkQVJHU1xuIyBjb21iaW5lIGFuZCByZXBsYWNlXG5maXg6ICd7ICRMSVRFUkFMIH0nIiwic291cmNlIjoiZCA9IGRpY3QoYT0xLCBiPTIpIn0=).

## `rewriters` is Top Level

Every ast-grep rule can have one `rewriters` at top level. The `rewriters` accepts a list of rewriter rules.

Every rewriter rule is like a regular ast-grep rule with `fix`. These are required fields for a rewriter rule.

* `id`: A unique identifier for the rewriter to be referenced in the `rewrite` transformation field.
* `rule`: A rule object to match the sub node.
* `fix`: A string to replace the matched sub node.

Rewriter rule can also have other fields like `transform` and `constraints`. However, fields like `severity` and `message` are not available in rewriter rules. Generally, only [Finding](/reference/yaml.html#finding) and [Patching](/reference/yaml.html#patching) fields are allowed in rewriter rules.

## Apply Multiple Rewriters

Note that the `rewrite` transformation field can accept multiple rewriters. This allows you to apply multiple rewriters to different sub nodes.

If the `source` meta variable contains multiple sub nodes, each sub node will be transformed by the corresponding rewriter that matches the sub node.

Suppose we have two rewriters to rewrite numbers and strings.

```yaml
rewriters:
- id: rewrite-int
  rule: {kind: integer}
  fix: integer
- id: rewrite-str
  rule: {kind: string}
  fix: string
```

We can apply both rewriters to the same source meta-variable.

```yaml
rule: {pattern: '[$$$LIST]' }
transform:
  NEW_VAR:
    rewrite:
      rewriters: [rewrite-num, rewrite-str]
      source: $$$LIST
```

In this case, the `rewrite-num` rewriter will be applied to the integer nodes in `$$$LIST`, and the `rewrite-str` rewriter will be applied to the string nodes in `$$$LIST`.

The produced `NEW_VAR` will contain the transformed nodes from both rewriters. [For example](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGljdCgkJCRBUkdTKSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJld3JpdGVyczpcbi0gaWQ6IHJld3JpdGUtaW50XG4gIHJ1bGU6IHtraW5kOiBpbnRlZ2VyfVxuICBmaXg6IGludGVnZXJcbi0gaWQ6IHJld3JpdGUtc3RyXG4gIHJ1bGU6IHtraW5kOiBzdHJpbmd9XG4gIGZpeDogc3RyaW5nXG5ydWxlOiB7cGF0dGVybjogJ1skJCRMSVNUXScgfVxudHJhbnNmb3JtOlxuICBORVdfVkFSOlxuICAgIHJld3JpdGU6XG4gICAgICByZXdyaXRlcnM6IFtyZXdyaXRlLWludCwgcmV3cml0ZS1zdHJdXG4gICAgICBzb3VyY2U6ICQkJExJU1RcbmZpeDogJE5FV19WQVIiLCJzb3VyY2UiOiJbMSwgJ2EnXSJ9), `[1, 'a']` will be transformed to `integer, string`.

:::tip Pro Tip
Using multiple rewriters can make you dynamically apply different rewriting logic to different sub nodes, based on the matching rules.
:::

In case multiple rewriters match the same sub node, only the matching rewriter that appears first in the `rewriters` list will be applied. Therefore, ***the order of rewriters in the `rewriters` list matters.***

## Use Alternative Joiner

By default, ast-grep will generate the new rewritten string by replacing the text in the matched sub nodes. But you can also specify an alternative joiner to join the transformed sub nodes via `joinBy` field.

```yaml
transform:
  NEW_VAR:
    rewrite:
      rewriters: [rewrite-num, rewrite-str]
      source: $$$LIST
      joinBy: ' + '
```

This will transform `1, 2, 3` to `integer + integer + integer`.

## Philosophy behind Rewriters

You can see a more detailed design philosophy, *Find and Patch*, behind rewriters in [this page](/advanced/find-n-patch.html).

---

---
url: /guide/tooling-overview.md
---
# Command Line Tooling Overview

## Overview

ast-grep's tooling supports multiple stages of your development. Here is a list of the tools and their purpose:

* To run an ad-hoc query and apply rewrite: `ast-grep run`.
* Routinely check your codebase: `ast-grep scan`.
* Generate ast-grep's scaffolding files: `ast-grep new`.
* Develop new ast-grep rules and test them: `ast-grep test`.
* Start Language Server for editor integration: `ast-grep lsp`.

We will walk through some important features that are common to these commands.

## Interactive Mode

ast-grep by default will output the results of your query at once in your terminal which is useful to have a quick glance at the result. But sometimes you will need to scrutinize every result one by one to refine you pattern query or to avoid bad cases for edge-case code.

You can use the `--interactive` flag to open an interactive mode. This will allow you to select which results you want to apply the rewrite to. This mode is inspired by [fast-mod](https://github.com/facebookincubator/fastmod).

Screenshot of interactive mode.
![interactive](/image/interactive.jpeg)

Pressing `y` will accept the rewrite, `n` will skip it, `e` will open the file in your editor, and `q` will quit the interactive mode.

Example:

```bash
ast-grep scan --interactive
```

## JSON Mode

Composability is a key perk of command line tooling. ast-grep is no exception.

`--json` will output results in JSON format. This is useful to pipe the results to other tools. For example, you can use [jq](https://stedolan.github.io/jq/) to extract information from the results and render it in [jless](https://jless.io/).

```bash
ast-grep run -p 'Some($A)' -r 'None' --json | jq '.[].replacement' | jless
```

The format of the JSON output is an array of match objects.

```json
[
  {
    "text": "import",
    "range": {
      "byteOffset": {
        "start": 66,
        "end": 72
      },
      "start": {
        "line": 3,
        "column": 2
      },
      "end": {
        "line": 3,
        "column": 8
      }
    },
    "file": "website/src/vite-env.d.ts",
    "replacement": "require",
    "language": "TypeScript"
  }
]
```

See [JSON mode doc](/guide/tools/json.html) for more detailed explanation and examples.

## Run One Single Query or One Single Rule

You can also use ast-grep to explore a proper pattern for your query. There are two ways to try your pattern or rule.
For testing one pattern, you can use `ast-grep run` command.

```bash
ast-grep run -p 'YOUR_PATTERN' --debug-query
```

The `--debug-query` option will output the tree-sitter ast of the query.

To test one single rule, you can use `ast-grep scan -r`.

```bash
ast-grep scan -r path/to/your/rule.yml
```

It is useful to test one rule in isolation.

## Parse Code from StdIn

ast-grep's `run` and `scan` commands also support searching and replacing code from [standard input (StdIn)](https://www.wikiwand.com/en/Standard_streams).
This mode is enabled by passing command line argument flag `--stdin`.
You can use bash's [pipe operator](https://linuxhint.com/bash_pipe_tutorial/) `|` to instruct ast-grep to read from StdIn.

### Example: Simple Web Crawler

Let's see an example in action. Combining with `curl`, `ast-grep` and `jq`, we can build a [simple web crawler](https://twitter.com/trevmanz/status/1671572111582978049) on command line. The command below uses `curl` to fetch the HTML source of SciPy conference website, and then uses `ast-grep` to parse and extract relevant information as JSON from source, and finally uses `jq` to transform our matching results.

```bash
curl -s https://schedule2021.scipy.org/2022/conference/  |
  ast-grep -p '<div $$$> $$$ <i>$AUTHORS</i> </div>' --lang html --json --stdin |
  jq '
    .[]
    | .metaVariables
    | .single.AUTHORS.text'
```

The command above will produce a list of authors from the SciPy 2022 conference website.

:::details JSON output of the author list

```json
"Ben Blaiszik"
"Qiming Sun"
"Max Jones"
"Thomas J. Fan"
"Sebastian Bichelmaier"
"Cliff Kerr"
...
```

:::

With this feature, even if your preferred language does not have native bindings for ast-grep, you can still parse code from standard input (StdIn) to use ast-grep programmatically from the command line.

You can invoke `ast-grep`, the command-line interface binary, as a subprocess to search and replace code.

### Caveats

**StdIn mode has several restrictions**, though:

* It conflicts with `--interactive` mode, which reads user responses from StdIn.
* For the `run` command, you must specify the language of the StdIn code with `--lang` or `-l` flag. For example: `echo "print('Hello world')" | ast-grep run --lang python`. This is because ast-grep cannot infer code language without file extension.
* Similarly, you can only `scan` StdIn code against *one single rule*, specified by `--rule` or `-r` flag. The rule must match the language of the StdIn code. For example: `echo "print('Hello world')" | ast-grep scan --rule "python-rule.yml"`

### Enable StdIn Mode

**All the following conditions** must be met to enable StdIn mode:

1. The command line argument flag `--stdin` is passed.
2. ast-grep is **NOT** running inside a [tty](https://github.com/softprops/atty). If you are using a terminal emulator, ast-grep will usually run in a tty if invoked directly from CLI.

The first condition is quite self explanatory. However, it should be noted that many cases are not tty, for example:

* ast-grep is invoked by other program as subprocess.
* ast-grep is running inside [GitHub Action](https://github.com/actions/runner/issues/241).
* ast-grep is used as the second program of a bash pipe `|`.

So you have to use `--stdin` to avoid unintentional StdIn mode and unexpected error.

:::danger Running ast-grep in tty with --stdin
ast-grep will hang there if you run it in a tty terminal session with `--stdin` flag, until you type in some text and send EOF signal (usually `Ctrl-D`).
:::

#### Bonus Example

Here is a bonus example to use [fzf](https://github.com/junegunn/fzf/blob/master/ADVANCED.md#using-fzf-as-interactive-ripgrep-launcher) as interactive ast-grep launcher.

```bash
SG_PREFIX="ast-grep run --color=always -p "
INITIAL_QUERY="${*:-}"
: | fzf --ansi --disabled --query "$INITIAL_QUERY" \
    --bind "start:reload:$SG_PREFIX {q}" \
    --bind "change:reload:sleep 0.1; $SG_PREFIX {q} || true" \
    --delimiter : \
    --preview 'bat --color=always {1} --highlight-line {2}' \
    --preview-window 'up,60%,border-bottom,+{2}+3/3,~3' \
    --bind 'enter:become(vim {1} +{2})'
```

## Editor Integration

See the [editor integration](/guide/tools/editors.md) doc page.

## Shell Completions

ast-grep comes with shell autocompletion scripts. You can generate a shell script and eval it when your shell starts up.
The script will enable you to smoothly complete `ast-grep` command's options by `tab`bing.

This command will instruct ast-grep  to generate shell completion script:

```shell
ast-grep completions <SHELL>
```

`<SHELL>` is an optional argument and can be one of the `bash`, `elvish`, `fish`, `powershell` and `zsh`. If shell is not specified, ast-grep will infer the correct shell from environment variable like `$SHELL`.

The exact steps required to enable autocompletion will vary by shell. For instructions, see the [Poetry](https://python-poetry.org/docs/#installing-with-the-official-installer) or [ripgrep](https://github.com/BurntSushi/ripgrep/blob/master/FAQ.md#complete) documentation.

### Example

If you are using zsh, add this line to your `~/.zshrc`.

```shell
eval "$(ast-grep completions)"
```

## Use ast-grep in GitHub Action

If you want to automate [ast-grep linting](https://github.com/marketplace/actions/ast-grep-gh-action) in your repository, you can use [GitHub Action](https://github.com/features/actions), a feature that lets you create custom workflows for different events.

For example, you can run ast-grep linting every time you push a new commit to your main branch.

To use ast-grep in GitHub Action, you need to [set up a project](/guide/scan-project.html) first. You can do this by running `ast-grep new` in your terminal, which will guide you through the process of creating a configuration file and a rules file.

Next, you need to create a workflow file for GitHub Action. This is a YAML file that defines the steps and actions that will be executed when a certain event occurs. You can create a workflow file named `ast-grep.yml` under the `.github/workflows/` folder in your repository, with the following content:

```yml
on: [push]

jobs:
  sg-lint:
    runs-on: ubuntu-latest
    name: Run ast-grep lint
    steps:
      - name: Checkout
        uses: actions/checkout@v4
      - name: ast-grep lint step
        uses: ast-grep/action@v1.4
```

This workflow file tells GitHub Action to run ast-grep linting on every push event, using the latest Ubuntu image and the official ast-grep action.
The action will check out your code and run [`ast-grep scan`](/reference/cli.html#ast-grep-scan) on it, reporting any errors or warnings.

That's it! You have successfully set up ast-grep linting in GitHub Action. Now, every time you push a new commit to your main branch, GitHub Action will automatically run ast-grep linting and show you the results. You can see an example of how it looks like below.

![image](https://github.com/ast-grep/action/assets/2883231/52fe5914-5e43-4478-a7b2-fb0399f61dee)

For more information, you can refer to the [ast-grep/action](https://github.com/ast-grep/action) repository, where you can find more details and options for using ast-grep in GitHub Action.

## Colorful Output

The output of ast-grep is exuberant and beautiful! But it is not always desired for colorful output.
You can use `--color never` to disable ANSI color in the command line output.

---

---
url: /guide/tools/editors.md
---
# Editor Integration

ast-grep is a **command line tool** for structural search/replace. But it can be readily integrated into your editors and streamline your workflow.

This page introduces several **editors** that has ast-grep support.

## VSCode

ast-grep has an official [VSCode extension](https://marketplace.visualstudio.com/items?itemName=ast-grep.ast-grep-vscode#overview) in the market place.

To get a feel of what it can do, see the introduction on YouTube!

### Features

The ast-grep VSCode is an extension to bridge the power of ast-grep and the beloved editor VSCode.
It includes two parts:

* a UI for ast-grep CLI and
* a client for ast-grep LSP.

:::tip Requirement
You need to [install ast-grep CLI](/guide/quick-start.html#installation) locally and optionally [set up a linting project](/guide/scan-project.html).
:::

### Structural Search

Use [pattern](https://ast-grep.github.io/guide/pattern-syntax.html) to structural search your codebase.

| Feature         | Screenshot                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| Search Pattern  |      |
| Search YAML     |      |
| Search In Folder|   |

### Structural Replace

Use pattern to [replace](https://ast-grep.github.io/guide/rewrite-code.html) matching code.

| Feature         | Screenshot                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| Replace Preview |              |
| Commit Replace  |      |

### Diagnostics and Code Action

*Require LSP setup*

Code linting and code actions require [setting up `sgconfig.yml`](https://ast-grep.github.io/guide/scan-project.html) in your workspace root.

| Feature         | Screenshot                                                                                                  |
| --------------- | ----------------------------------------------------------------------------------------------------------- |
| Code Linting    |                |

### FAQs

#### Why LSP diagnostics are not working?

You need several things to set up LSP diagnostics:

1. [Install](/guide/quick-start.html#installation) ast-grep CLI. Make sure it is accessible in VSCode editor.
2. [Set up a linting project](/guide/scan-project.html) in your workspace root. The `sgconfig.yml` file is required for LSP diagnostics.
3. The LSP server by default is started in the workspace root. Make sure the `sgconfig.yml` is in the workspace root.

#### Why ast-grep VSCode cannot find the CLI?

The extension has a different environment from the terminal. You need to make sure the CLI is accessible in the extension environment. For example, if the CLI is installed in a virtual environment, you need to activate the virtual environment in the terminal where you start VSCode.

Here are a few ways to make the CLI accessible:

1. Install the CLI globally.
2. Specify the CLI path in the extension settings `astGrep.serverPath`.
3. Check if VSCode has the same `PATH` as the terminal.

#### Project Root Detection

By default, ast-grep will only start in the workspace root. If you want to start ast-grep in a subfolder, you can specify the `configPath` in the extension settings.
The `configPath` is the path to the `sgconfig.yml` file and is relative to the workspace root.

#### Schema Validation

When writing your own `rule.yml` file, you can use schema validation to get quick feedback on whether your file is structured properly.

1. Add the following line to the top of your file:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/ast-grep/ast-grep/main/schemas/rule.json
```

2. Install a VSCode extension that supports schema validation for yaml files. For example, [YAML by Red Hat](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml).

![Schema Validation](/image/schema-validation.png)
After reloading the VSCode window, you should see red underlines for any errors in your `rule.yml` file, along with autocompletions and tooltips on hover. In VSCode you can typically use \[Ctrl] + \[Space] to see the available autocompletions.

## Neovim

### nvim-lspconfig

The recommended setup is using [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig).

```lua
require('lspconfig').ast_grep.setup({
  -- these are the default options, you only need to specify
  -- options you'd like to change from the default
  cmd = { 'ast-grep', 'lsp' },
  filetypes = { "c", "cpp", "rust", "go", "java", "python", "javascript", "typescript", "html", "css", "kotlin", "dart", "lua" },
  root_dir = require('lspconfig.util').root_pattern('sgconfig.yaml', 'sgconfig.yml')
})
```

### coc.nvim

Please see [coc-ast-grep](https://github.com/yaegassy/coc-ast-grep)

You need to have coc.nvim installed for this extension to work. e.g. vim-plug:

```vim
Plug 'yaegassy/coc-ast-grep', {'do': 'yarn install --frozen-lockfile'}
```

### telescope.nvim

[telescope-sg](https://github.com/Marskey/telescope-sg) is the ast-grep picker for telescope.nvim.

Usage:

```vim
Telescope ast_grep
```

[telescope-ast-grep.nvim](https://github.com/ray-x/telescope-ast-grep.nvim) is an alternative plugin that provides ast-grep functionality enhancements.

### grug-far.nvim

[grug-far.nvim](https://github.com/MagicDuck/grug-far.nvim) has ast-grep search engine support. It allows for both live searching as you type and replacing.

Usage:

```vim
:lua require('grug-far').grug_far({ engine = 'astgrep' })
```

or swap to `astgrep` engine while running with the `Swap Engine` action.

## Emacs

### ast-grep.el

[ast-grep.el](https://github.com/SunskyXH/ast-grep.el) is an emacs package for searching code using ast-grep with completing-read interface or consult.

You can install via `straight.el`

```elisp
(straight-use-package '(ast-grep :type git :host github :repo "SunskyXH/ast-grep.el"))
```

Or if you are using doomemacs, add to your `packages.el`

```elisp
(package! ast-grep :recipe (:host github :repo "SunskyXH/ast-grep.el"))
```

## LSP Server

Currently ast-grep support these LSP capabilities:

### Server capability

* [publish diagnostics](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_publishDiagnostics)
* [Fix diagnostic code action](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_publishCodeAction)

### Client requirements

* [textDocument/didOpen](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didOpen)
* [textDocument/didChange](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didChange)
* [textDocument/didClose](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_didClose)

### Configuration

ast-grep does not have LSP configuration, except that ast-grep LSP requires `sgconfig.yml` in the project root.

You can also specify the configuration file path via command line:

```bash
ast-grep lsp -c <configPath>
```

## More Editors...

More ast-grep editor integration will be supported by the community!
Your contribution is warmly welcome.

---

---
url: /guide/tools/json.md
---
# JSON Mode

Composability is a key perk of command line tooling. ast-grep is no exception.

`--json` will output results in JSON format. This is useful to pipe the results to other tools.

**Example:**

```bash
ast-grep run -p 'Some($A)' -r 'None' --json
```

## Output Data Structure

The format of the JSON output is an array of match objects. Below is an example of a match object generated from the command above.

```json
[
  {
    "text": "Some(matched)",
    "range": {
      "byteOffset": { "start": 10828, "end": 10841 },
      "start": { "line": 303, "column": 2 },
      "end": { "line": 303, "column": 15 }
    },
    "file": "crates/config/src/rule/mod.rs",
    "lines": "  Some(matched)",
    "replacement": "None",
    "replacementOffsets": { "start": 10828, "end": 10841 },
    "language": "Rust",
    "metaVariables": {
      "single": {
        "A": {
          "text": "matched",
          "range": {
            "byteOffset": { "start": 10833, "end": 10840 },
            "start": { "line": 303, "column": 7 },
            "end": { "line": 303, "column": 14 }
          }
        }
      },
      "multi": {},
      "transformed": {}
    }
  }
]
```

### Match Object Type

Below is the equivalent TypeScript type definition of the match object.

```typescript
interface Match {
  text: string
  range: Range
  file: string // relative path to the file
  // the surrounding lines of the match.
  // It can be more than one line if the match spans multiple ones.
  lines: string
  // optional replacement if the match has a replacement
  replacement?: string
  replacementOffsets?: ByteOffset
  metaVariables?: MetaVariables // optional metavars generated in the match
}

interface Range {
  byteOffset: ByteOffset
  start: Position
  end: Position
}
// UTF-8 encoded byte offset
interface ByteOffset {
  start: number // start is inclusive
  end: number   // end is exclusive
}
interface Position {
  line: number   // zero-based line number
  column: number // zero-based column number
}

// See Pattern doc
interface MetaVariables {
  single: Record<String, MetaVar>
  multi: Record<String, MetaVar[]>
  transformed: Record<String, String> // See Rewrite doc
}
interface MetaVar {
  text: string
  range: Range
}
```

For more information about `MetaVariables` and `transformed` fields, see the [Pattern](/guide/pattern-syntax.html#meta-variable) and [Rewrite](/guide/rewrite/transform.html) documentation.

If you are using [lint rule](/guide/project/lint-rule.html) to find matches, the generated match objects will have several more fields.

```typescript
interface RuleMatch extends Match {
  ruleId: string
  severity: Severity
  note?: string
  message: string
}

enum Severity {
  Error = "error",
  Warning = "warning",
  Info = "info",
  Hint = "hint",
}
```

:::tip line, column, and byte offset are zero-based
The `line`, `column`, and `byteOffset` fields are zero-based. This means that the first line, column, and byte offset are 0, not 1.
The design is consistent with the [LSP](https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#position) and [tree-sitter](https://tree-sitter.github.io/tree-sitter/using-parsers#syntax-nodes) specifications.

If you need 1-based numbers, you can use `jq` to transform the output.
:::

## Consuming JSON output

ast-grep embraces the Unix philosophy of composability. The `--json` flag is designed to make it easy to pipe the results to other tools.

For example, you can use [jq](https://stedolan.github.io/jq/) to extract information from the results and render it in [jless](https://jless.io/).

```bash
ast-grep run -p 'Some($A)' -r 'None' --json | jq '.[].replacement' | jless
```

You can also see [an example](https://github.com/ast-grep/ast-grep/issues/1232#issuecomment-2181747911) of using `--json` flag in Vim's QuickFix window.

## Output Format

By default, ast-grep prints the matches in a JSON array that is formatted with indentation and line breaks.
`--json` is equivalent to `--json=pretty`. This makes it easy to read the output by humans.
However, this might not be suitable for other programs that need to process the output from ast-grep. For example, if there are too many matches, the JSON array might be [too large to fit in memory](https://www.wikiwand.com/en/Out_of_memory).

To avoid this problem, you can use the `--json=stream` option when running ast-grep. This option will make ast-grep print each match as a separate JSON object, followed by a newline character. This way, you can stream the output to other programs that can read one object per line and parse it accordingly.

The output of `--json=stream` looks like below:

```
$ ast-grep -p pattern --json=stream
{"text":"Some(matched)", ... }
{"text":"Some(matched)", ... }
{"text":"Some(matched)", ... }
```

You can read the output line by line and process it accordingly.

`--json` accepts one of the following values: `pretty`, `stream`, or `compact`.

:::danger `--json=stream` requires the equal sign
You have to use `--json=<STYLE>` syntax when passing value to the json flag.
A common gotcha is missing the equal sign.
`--json stream` is parsed as `--json=pretty stream` and `stream` is parsed as a directory.
Only `--json=stream` will work as a key-value pair.
:::

---

---
url: /guide/api-usage.md
---
# API Usage

## ast-grep as Library

ast-grep allows you to craft complicated rules, but it is not easy to do arbitrary AST manipulation.

For example, you may struggle to:

* replace a list of nodes individually, based on their content
* replace a node conditionally, based on its content and surrounding nodes
* count the number or order of nodes that match a certain pattern
* compute the replacement string based on the matched nodes

To solve these problems, you can use ast-grep's programmatic API! You can freely inspect and generate text patches based on syntax trees, using popular programming languages!

:::tip
Applying ast-grep's `fix` using JS/Python API is still experimental. See [this issue](https://github.com/ast-grep/ast-grep/issues/1172) for more information.
:::

## Language Bindings

ast-grep provides support for these programming languages:

* **JavaScript:** Powered by napi.rs, ast-grep's JavaScript API is the most robust and reliable. [Explore JavaScript API](/guide/api-usage/js-api.html)

* **Python:** ast-grep's PyO3 interface is the latest addition to climb the syntax tree! [Discover Python API](/guide/api-usage/py-api.html)

* **Rust:** ast-grep's Rust API is the most efficient way, but also the most challenging way, to use ast-grep. You can refer to [ast\_grep\_core](https://docs.rs/ast-grep-core/latest/ast_grep_core/) if you are familiar with Rust.

## Why and When to use API?

ast-grep's API is designed to solve the problems that are hard to express in ast-grep's rule language.

ast-grep's rule system is deliberately simple and not as powerful as a programming language.
Other similar rewriting/query tools have complex features like conditional, loop, filter or function call.
These features are hard to learn and use, and they cannot perform computation as well as a general purpose programming language.

So ast-grep chooses to have a simple rule system that is easy to learn and use. But it also has its limitations. The API is created to overcome these limitations.

If your code transformation requires complex logic, or if you need to change code that has no parser library in JavaScript or Python, ast-grep API is a good option to achieve your goal without writing a lot of complicated rules.

---

---
url: /guide/api-usage/js-api.md
---
# JavaScript API

Powered by [napi.rs](https://napi.rs/), ast-grep's JavaScript API enables you to write JavaScript to programmatically inspect and change syntax trees.

ast-grep's JavaScript API design is pretty stable now. No major breaking changes are expected in the future.

To try out the JavaScript API, you can use the [code sandbox](https://codesandbox.io/p/sandbox/ast-grep-napi-hhx3tj) here.

## Installation

First, install ast-grep's napi package.

::: code-group

```bash[npm]
npm install --save @ast-grep/napi
```

```bash[pnpm]
pnpm add @ast-grep/napi
```

:::

Now let's explore ast-grep's API!

## Core Concepts

The core concepts in ast-grep's JavaScript API are:

* `SgRoot`: a class representing the whole syntax tree
* `SgNode`: a node in the syntax tree

:::tip Make AST like a DOM tree!
Using ast-grep's API is like using [jQuery](https://jquery.com/). You can use `SgNode` to traverse the syntax tree and collect information from the nodes.

Remember your old time web programming?
:::

A common workflow to use ast-grep's JavaScript API is:

1. Get a syntax tree object `SgRoot` from string by calling a language's `parse` method
2. Get the root node of the syntax tree by calling `ast.root()`
3. `find` relevant nodes by using patterns or rules
4. Collect information from the nodes

**Example:**

```js{4-7}
import { parse, Lang } from '@ast-grep/napi';

let source = `console.log("hello world")`
const ast = parse(Lang.JavaScript, source)  // 1. parse the source
const root = ast.root()                     // 2. get the root
const node = root.find('console.log($A)')   // 3. find the node
node.getMatch('A').text()                   // 4. collect the info
// "hello world"
```

### `SgRoot`

`SgRoot` represents the syntax tree of a source string.

We can import the `Lang` enum from the `@ast-grep/napi` package and call the  `parse` function to transform string.

```js{4}
import { Lang, parse } from '@ast-grep/napi';

const source = `console.log("hello world")`
const ast = parse(Lang.JavaScript, source)
```

The `SgRoot` object has a `root` method that returns the root `SgNode` of the AST.

```js
const root = ast.root() // root is an instance of SgNode
```

### `SgNode`

`SgNode` is the main interface to view and manipulate the syntax tree.

It has several jQuery like methods for us to search, filter and inspect the AST nodes we are interested in.

```js
const log = root.find('console.log($A)') // search node
const arg = log.getMatch('A') // get matched variable
arg.text() // "hello world"
```

Let's see its details in the following sections!

## Search

You can use `find` and `findAll` to search for nodes in the syntax tree.

* `find` returns the first node that matches the pattern or rule.
* `findAll` returns an array of nodes that match the pattern or rule.

```ts
// search
class SgNode {
  find(matcher: string): SgNode | null
  find(matcher: number): SgNode | null
  find(matcher: NapiConfig): SgNode | null
  findAll(matcher: string): Array<SgNode>
  findAll(matcher: number): Array<SgNode>
  findAll(matcher: NapiConfig): Array<SgNode>
}
```

Both `find` and `findAll` are overloaded functions. They can accept either string, number or a config object.
The argument is called `Matcher` in ast-grep JS.

### Matcher

A `Matcher` can be one of the three types: `string`, `number` or `object`.

* `string` is parsed as a [pattern](/guide/pattern-syntax.html). e.g. `'console.log($A)'`

* `number` is interpreted as the node's kind. In tree-sitter, an AST node's type is represented by a number called kind id. Different syntax node has different kind ids. You can convert a kind name like `function` to the numeric representation by calling the `kind` function. e.g. `kind(Lang.JavaScript, 'function')`.

* A `NapiConfig` has a similar type of [config object](/reference/yaml.html). See details below.

```ts
// basic find example
root.find('console.log($A)')    // returns SgNode of call_expression
let l = Lang.JavaScript         // calling kind function requires Lang
const kind = kind(l, 'string')  // convert kind name to kind id number
root.find(kind)                 // returns SgNode of string
root.find('notExist')           // returns null if not found

// basic find all example
const nodes = root.findAll('function $A($$$) {$$$}')
Array.isArray(nodes)     // true, findAll returns SgNode
nodes.map(n => n.text()) // string array of function source
const empty = root.findAll('not exist') // returns []
empty.length === 0 // true

// find i.e. `console.log("hello world")` using a NapiConfig
const node = root.find({
  rule: {
    pattern: "console.log($A)"
  },
  constraints: {
    A: { regex: "hello" }
  }
})
```

Note, `find` returns `null` if no node is found. `findAll` returns an empty array if nothing matches.

## Match

Once we find a node, we can use the following methods to get meta variables from the search.

The `getMatch` method returns the single node that matches the [single meta variable](/guide/pattern-syntax.html#meta-variable).

And the `getMultipleMatches` returns an array of nodes that match the [multi meta variable](/guide/pattern-syntax.html#multi-meta-variable).

```ts
// search
export class SgNode {
  getMatch(m: string): SgNode | null
  getMultipleMatches(m: string): Array<SgNode>
}
```

**Example:**

```ts{7,11,15,16}
const src = `
console.log('hello')
logger('hello', 'world', '!')
`
const root = parse(Lang.JavaScript, src).root()
const node = root.find('console.log($A)')
const arg = node.getMatch("A") // returns SgNode('hello')
arg !== null // true, node is found
arg.text() // returns 'hello'
// returns [] because $A and $$$A are different
node.getMultipleMatches('A')

const logs = root.find('logger($$$ARGS)')
// returns [SgNode('hello'), SgNode(','), SgNode('world'), SgNode(','), SgNode('!')]
logs.getMultipleMatches("ARGS")
logs.getMatch("A") // returns null
```

## Inspection

The following methods are used to inspect the node.

```ts
// node inspection
export class SgNode {
  range(): Range
  isLeaf(): boolean
  kind(): string
  text(): string
}
```

**Example:**

```ts{3}
const ast = parse(Lang.JavaScript, "console.log('hello world')")
root = ast.root()
root.text() // will return "console.log('hello world')"
```

Another important method is `range`, which returns two `Pos` object representing the start and end of the node.

One `Pos` contains the line, column, and offset of that position. All of them are 0-indexed.

You can use the range information to locate the source and modify the source code.

```ts{1}
const rng = node.range()
const pos = rng.start // or rng.end, both are `Pos` objects
pos.line // 0, line starts with 0
pos.column // 0, column starts with 0
rng.end.index // 17, index starts with 0
```

## Refinement

You can also filter nodes after matching by using the following methods.

This is dubbed as "refinement" in the documentation. Note these refinement methods only support using `pattern` at the moment.

```ts
export class SgNode {
  matches(m: string): boolean
  inside(m: string): boolean
  has(m: string): boolean
  precedes(m: string): boolean
  follows(m: string): boolean
}
```

**Example:**

```ts
const node = root.find('console.log($A)')
node.matches('console.$METHOD($B)') // true
```

## Traversal

You can traverse the tree using the following methods, like using jQuery.

```ts
export class SgNode {
  children(): Array<SgNode>
  field(name: string): SgNode | null
  parent(): SgNode | null
  child(nth: number): SgNode | null
  ancestors(): Array<SgNode>
  next(): SgNode | null
  nextAll(): Array<SgNode>
  prev(): SgNode | null
  prevAll(): Array<SgNode>
}
```

## Fix code

`SgNode` is immutable so it is impossible to change the code directly.

However, `SgNode` has a `replace` method to generate an `Edit` object. You can then use the `commitEdits` method to apply the changes and generate new source string.

```ts
interface Edit {
  /** The start position of the edit */
  startPos: number
  /** The end position of the edit */
  endPos: number
  /** The text to be inserted */
  insertedText: string
}

class SgNode {
  replace(text: string): Edit
  commitEdits(edits: Edit[]): string
}
```

**Example**

```ts{3,4}
const root = parse(Lang.JavaScript, "console.log('hello world')").root()
const node = root.find('console.log($A)')
const edit = node.replace("console.error('bye world')")
const newSource = node.commitEdits([edit])
// "console.error('bye world')"
```

Note, `console.error($A)` will not generate `console.error('hello world')` in JavaScript API unlike the CLI. This is because using the host language to generate the replacement string is more flexible.

:::warning
Metavariable will not be replaced in the `replace` method. You need to create a string using `getMatch(var_name)` by using JavaScript.
:::

See also [ast-grep#1172](https://github.com/ast-grep/ast-grep/issues/1172)

## Use Other Language

To access other languages, you will need to use `registerDynamicLanguage` function and probably `@ast-grep/lang-*` package.
This is an experimental feature and the doc is not ready yet. Please refer to the [repo](https://github.com/ast-grep/langs) for more information.

If you are interested in using other languages, please let us know by creating an issue.

---

---
url: /guide/api-usage/py-api.md
---
# Python API

ast-grep's Python API is powered by [PyO3](https://pyo3.rs/).
You can write Python to programmatically inspect and change syntax trees.

To try out ast-grep's Python API, you can use the [online colab notebook](https://colab.research.google.com/drive/1nVT6rQKRIPv0TsKpCv5uD-Zuw-lUC67A?usp=sharing).

## Installation

ast-grep's Python library is distributed on PyPI. You can install it with pip.

```bash
pip install ast-grep-py
```

## Core Concepts

The core concepts in ast-grep's Python API are:

* `SgRoot`: a class to parse a string into a syntax tree
* `SgNode`: a node in the syntax tree

:::tip Make AST like a XML/HTML doc!
Using ast-grep's API is like [web scraping](https://opensource.com/article/21/9/web-scraping-python-beautiful-soup) using [beautiful soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) or [pyquery](https://pyquery.readthedocs.io/en/latest/). You can use `SgNode` to traverse the syntax tree and collect information from the nodes.
:::

A common workflow to use ast-grep's Python API is:

1. Parse a string into a syntax tree by using `SgRoot`
2. Get the root node of the syntax tree by calling `root.root()`
3. `find` relevant nodes by using patterns or rules
4. Collect information from the nodes

**Example:**

```python{3-6}
from ast_grep_py import SgRoot

root = SgRoot("print('hello world')", "python") # 1. parse
node = root.root()                              # 2. get root
print_stmt = node.find(pattern="print($A)")     # 3. find
print_stmt.get_match('A').text()                # 4. collect information
# 'hello world'
```

### `SgRoot`

The `SgRoot` class has the following signature:

```python
class SgRoot:
    def __init__(self, src: str, language: str) -> None: ...
    def root(self) -> SgNode: ...
```

`__init__` takes two arguments: the first argument is the source code string, and the second argument is the language name.
`root` returns the root node of the syntax tree, which is an instance of `SgNode`.

**Example:**

```python
root = SgRoot("print('hello world')", "python") # 1. parse
node = root.root()                              # 2. get root
```

The code above parses the string `print('hello world')` into a syntax tree, and gets the root node of the syntax tree.

The root node can be used to find other nodes in the syntax tree.

### `SgNode`

`SgNode` is the most important class in ast-grep's Python API. It provides methods to inspect and traverse the syntax tree.
The following sections will introduce several methods in `SgNode`.

**Example:**

```python
node = root.root()
string = node.find(kind="string")
assert string # assume we can find a string node in the source
print(string.text())
```

## Search

You can use `find` and `find_all` to search for nodes in the syntax tree.

* `find` returns the first node that matches the pattern or rule.
* `find_all` returns a list of nodes that match the pattern or rule.

```python
# Search
class SgNode:
    @overload
    def find(self, **kwargs: Unpack[Rule]) -> Optional[SgNode]: ...
    @overload
    def find_all(self, **kwargs: Unpack[Rule]) -> List[SgNode]: ...
    @overload
    def find(self, config: Config) -> Optional[SgNode]: ...
    @overload
    def find_all(self, config: Config) -> List[SgNode]: ...
```

`find` has two overloads: one takes keyword arguments of [`Rule`](/reference/api.html#rule), and the other takes a [`Config`](/reference/api.html#config) object.

### Search with Rule

Using keyword arguments rule is the most straightforward way to search for nodes.

The argument name is the key of a rule, and the argument value is the rule's value.
You can passing multiple keyword arguments to `find` to search for nodes that match **all** the rules.

```python
root = SgRoot("print('hello world')", "python")
node = root.root()
node.find(pattern="print($A)") # will return the print function call
node.find(kind="string") # will return the string 'hello world'
# below will return print function call because it matches both rules
node.find(pattern="print($A)", kind="call")
# below will return None because the pattern cannot be a string literal
node.find(pattern="print($A)", kind="string")

strings = node.find_all(kind="string") # will return [SgNode("hello world")]
assert len(strings) == 1
```

### Search with Config

You can also use a `Config` object to search for nodes. This is similar to directly use YAML in the command line.

The main difference between using `Config` and using `Rule` is that `Config` has more options to control the search behavior, like [`constraints`](/guide/rule-config.html#constraints) and [`utils`](/guide/rule-config/utility-rule.html).

```python
# will find a string node with text 'hello world'
root.root().find({
  "rule": {
    "pattern": "print($A)",
  },
  "constraints": {
    "A": { "regex": "hello" }
  }
})
# will return None because constraints are not satisfied
root.root().find({
  "rule": {
    "pattern": "print($A)",
  },
  "constraints": {
    "A": { "regex": "no match" }
  }
})
```

## Match

Once we find a node, we can use the following methods to get meta variables from the search.

The `get_match` method returns the single node that matches the [single meta variable](/guide/pattern-syntax.html#meta-variable).

And the `get_multiple_matches` returns a list of nodes that match the [multi meta variable](/guide/pattern-syntax.html#multi-meta-variable).

```python
class SgNode:
    def get_match(self, meta_var: str) -> Optional[SgNode]: ...
    def get_multiple_matches(self, meta_var: str) -> List[SgNode]: ...
    def __getitem__(self, meta_var: str) -> SgNode: ...
```

**Example:**

```python{7,11,15,16}
src = """
print('hello')
logger('hello', 'world', '!')
"""
root = SgRoot(src, "python").root()
node = root.find(pattern="print($A)")
arg = node.get_match("A") # returns SgNode('hello')
assert arg # assert node is found
arg.text() # returns 'hello'
# returns [] because $A and $$$A are different
node.get_multiple_matches("A")

logs = root.find(pattern="logger($$$ARGS)")
# returns [SgNode('hello'), SgNode(','), SgNode('world'), SgNode(','), SgNode('!')]
logs.get_multiple_matches("ARGS")
logs.get_match("A") # returns None
```

`SgNode` also supports `__getitem__` to get the match of single meta variable.

It is equivalent to `get_match` except that it will either return `SgNode` or raise a `KeyError` if the match is not found.

Use `__getitem__` to avoid unnecessary `None` checks when you are using a type checker.

```python
node = root.find(pattern="print($A)")
# node.get_match("A").text() # error: node.get_match("A") can be None
node["A"].text() # Ok
```

## Inspection

The following methods are used to inspect the node.

```python
# Node Inspection
class SgNode:
    def range(self) -> Range: ...
    def is_leaf(self) -> bool: ...
    def is_named(self) -> bool: ...
    def is_named_leaf(self) -> bool: ...
    def kind(self) -> str: ...
    def text(self) -> str: ...
```

**Example:**

```python
root = SgRoot("print('hello world')", "python")
node = root.root()
node.text() # will return "print('hello world')"
```

Another important method is `range`, which returns two `Pos` object representing the start and end of the node.

One `Pos` contains the line, column, and offset of that position. All of them are 0-indexed.

You can use the range information to locate the source and modify the source code.

```python
rng = node.range()
pos = rng.start # or rng.end, both are `Pos` objects
pos.line # 0, line starts with 0
pos.column # 0, column starts with 0
rng.end.index # 17, index starts with 0
```

## Refinement

You can also filter nodes after matching by using the following methods.

This is dubbed as "refinement" in the documentation. Note these refinement methods only support using `Rule`.

```python
# Search Refinement
class SgNode:
    def matches(self, **rule: Unpack[Rule]) -> bool: ...
    def inside(self, **rule: Unpack[Rule]) -> bool: ...
    def has(self, **rule: Unpack[Rule]) -> bool: ...
    def precedes(self, **rule: Unpack[Rule]) -> bool: ...
    def follows(self, **rule: Unpack[Rule]) -> bool: ...
```

**Example:**

```python
node = root.find(pattern="print($A)")
if node["A"].matches(kind="string"):
  print("A is a string")
```

## Traversal

You can traverse the tree using the following methods, like using pyquery.

```python
# Tree Traversal
class SgNode:
    def get_root(self) -> SgRoot: ...
    def field(self, name: str) -> Optional[SgNode]: ...
    def parent(self) -> Optional[SgNode]: ...
    def child(self, nth: int) -> Optional[SgNode]: ...
    def children(self) -> List[SgNode]: ...
    def ancestors(self) -> List[SgNode]: ...
    def next(self) -> Optional[SgNode]: ...
    def next_all(self) -> List[SgNode]: ...
    def prev(self) -> Optional[SgNode]: ...
    def prev_all(self) -> List[SgNode]: ...
```

## Fix code

`SgNode` is immutable so it is impossible to change the code directly.

However, `SgNode` has a `replace` method to generate an `Edit` object. You can then use the `commitEdits` method to apply the changes and generate new source string.

```python
class Edit:
    # The start position of the edit
    start_pos: int
    # The end position of the edit
    end_pos: int
    # The text to be inserted
    inserted_text: str

class SgNode:
    # Edit
    def replace(self, new_text: str) -> Edit: ...
    def commit_edits(self, edits: List[Edit]) -> str: ...
```

**Example**

```python
root = SgRoot("print('hello world')", "python").root()
node = root.find(pattern="print($A)")
edit = node.replace("logger.log('bye world')")
new_src = node.commit_edits([edit])
# "logger.log('bye world')"
```

Note, `logger.log($A)` will not generate `logger.log('hello world')` in Python API unlike the CLI. This is because using the host language to generate the replacement string is more flexible.

:::warning
Metavariable will not be replaced in the `replace` method. You need to create a string using `get_match(var_name)` by using Python.
:::

See also [ast-grep#1172](https://github.com/ast-grep/ast-grep/issues/1172)

---

---
url: /guide/api-usage/performance-tip.md
---
# Performance Tip for napi usage

Using `napi` to parse code and search for nodes [isn't always faster](https://medium.com/@hchan_nvim/benchmark-typescript-parsers-demystify-rust-tooling-performance-025ebfd391a3) than pure JavaScript implementations.

There are a lot of tricks to improve performance when using `napi`. The mantra is to *reduce FFI (Foreign Function Interface) calls between Rust and JavaScript*, and to *take advantage of parallel computing*.

## Prefer `parseAsync` over `parse`

`parseAsync` can take advantage of NodeJs' libuv thread pool to parse code in parallel threads. This can be faster than the sync version `parse` when handling a lot of code.

```ts
import { Lang, parse, parseAsync } from '@ast-grep/napi'
// only one thread parsing
const ast = parse(Lang.JavaScript, 'console.log("hello world")')
const root = ast.root()
// better, can use multiple threads
const ast2 = await parseAsync(Lang.JavaScript, 'console.log("hello world")')
const root2 = ast2.root()
```

This is especially useful when you are using ast-grep in bundlers where the main thread is busy with other CPU intensive tasks.

## Prefer `findAll` over manual traversal

One way to find all nodes that match a rule is to traverse the syntax tree manually and check each node against the rule. This is slow because it requires a lot of FFI calls between Rust and JavaScript during the traversal.

For example, the following code snippet finds all `member_expression` nodes in the syntax tree. Unfortunately, there are as many FFI calls as the tree node number in the recursion.

```ts
const root = sgroot.root()
function findMemberExpression(node: SgNode): SgNode[] {
  let ret: SgNode[] = []
  // `node.kind()` is a FFI call
  if (node.kind() === 'member_expression') {
    ret.push(node)
  }
  // `node.children()` is a FFI call
  for (let child of node.children()) {
    // recursion makes more FFI calls
    ret = ret.concat(findMemberExpression(child))
  }
  return ret
}
const nodes = findMemberExpression(root)
```

The equivalent code using `findAll` is much faster:

```ts
const root = sgroot.root()
// only call FFI `findAll` once
const nodes = root.findAll({kind: 'member_expression'})
```

> *One [success](https://x.com/hd_nvim/status/1767971906786128316) [story](https://x.com/sonofmagic95/status/1768433654404104555) on Twitter, as an example.*

## Prefer `findInFiles` when possible

If you have a lot of files to parse and want to maximize your programs' performance, ast-grep provides a `findInFiles` function that parses multiple files and searches relevant nodes in parallel Rust threads.

APIs we showed above all require parsing code in Rust and pass the `SgRoot` back to JavaScript.
This incurs foreign function communication overhead and only utilizes the single main JavaScript thread.
By avoiding Rust-JS communication overhead and utilizing multiple core computing,
`findInFiles` is much faster than finding files in JavaScript and then passing them to Rust as string.

The function signature of `findInFiles` is as follows:

```ts
export function findInFiles(
  /** the language to parse */
  lang: Lang,
  /** specify the file path and matcher */
  config: FindConfig,
  /** callback function for found nodes in a file */
  callback: (err: null | Error, result: SgNode[]) => void
): Promise<number>
```

`findInFiles` accepts a `FindConfig` object and a callback function.

`FindConfig` specifies both what file path to *parse* and what nodes to *search*.

`findInFiles` will parse all files matching paths and will call back the function with nodes matching the `matcher` found in the files as arguments.

### `FindConfig`

The `FindConfig` object specifies which paths to search code and what rule to match node against.

The `FindConfig` object has the following type:

```ts
export interface FindConfig {
  paths: Array<string>
  matcher: NapiConfig
}
```

The `path` field is an array of strings. You can specify multiple paths to search code. Every path in the array can be a file path or a directory path. For a directory path, ast-grep will recursively find all files matching the language.

The `matcher` is the same as `NapiConfig` stated above.

### Callback Function and Termination

The `callback` function is called for every file that have nodes that match the rule. The callback function is a standard node-style callback with the first argument as `Error` and second argument as an array of `SgNode` objects that match the rule.

The return value of `findInFiles` is a `Promise` object. The promise resolves to the number of files that have nodes that match the rule.

:::danger
`findInFiles` can return before all file callbacks are called due to NodeJS limitation.
See https://github.com/ast-grep/ast-grep/issues/206.
:::

If you have a lot of files and `findInFiles` prematurely returns, you can use the total files returned by `findInFiles` as a check point. Maintain a counter outside of `findInFiles` and increment it in callback. If the counter equals the total number, we can conclude all files are processed. The following code is an example, with core logic highlighted.

```ts:line-numbers {11,16-18}
type Callback = (t: any, cb: any) => Promise<number>
function countedPromise<F extends Callback>(func: F) {
  type P = Parameters<F>
  return async (t: P[0], cb: P[1]) => {
    let i = 0
    let fileCount: number | undefined = undefined
    // resolve will be called after all files are processed
    let resolve = () => {}
    function wrapped(...args: any[]) {
      let ret = cb(...args)
      if (++i === fileCount) resolve()
      return ret
    }
    fileCount = await func(t, wrapped as P[1])
    // not all files are processed, await `resolve` to be called
    if (fileCount > i) {
      await new Promise<void>(r => resolve = r)
    }
    return fileCount
  }
}
```

### Example

Example of using `findInFiles`

```ts
import { Lang, findInFiles } from '@ast-grep/napi'

let fileCount = await findInFiles(Lang.JavaScript, {
  paths: ['relative/path/to/code'],
  matcher: {
    rule: {kind: 'member_expression'}
  },
}, (err, n) => {
  if (err) {
    console.error(err)
    return
  }
  console.log(`Found ${n.length} nodes`)
  console.log(n[0].text())
})
```

---

---
url: /catalog.md
---
# Rule Catalog

Get confused what ast-grep is? This is a list of rewriting rule to inspire you!
Explore the power of ast-grep with these rewriting rules that can transform your code in seconds.

Feel free to join our [Discord](https://discord.gg/4YZjf6htSQ) channel or ask [Codemod AI](https://app.codemod.com/studio?ai_thread_id=new) to explain the rules for you line by line!

---

---
url: /catalog/c.md
---
# C

This page curates a list of example ast-grep rules to check and to rewrite C code.

:::tip C files can be parsed as Cpp
You can parse C code as Cpp to avoid rewriting similar rules. The [`languageGlobs`](/reference/sgconfig.html#languageglobs) option can force ast-grep to parse `.c` files as Cpp.
:::

## Match Function Call in C

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImMiLCJxdWVyeSI6InRlc3QoJCQkKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAkTSgkJCQpO1xuICAgIHNlbGVjdG9yOiBjYWxsX2V4cHJlc3Npb24iLCJzb3VyY2UiOiIjZGVmaW5lIHRlc3QoeCkgKDIqeClcbmludCBhID0gdGVzdCgyKTtcbmludCBtYWluKCl7XG4gICAgaW50IGIgPSB0ZXN0KDIpO1xufSJ9)

### Description

One of the common questions of ast-grep is to match function calls in C.

A plain pattern like `test($A)` will not work. This is because [tree-sitter-c](https://github.com/tree-sitter/tree-sitter-c)
parse the code snippet into `macro_type_specifier`, see the [pattern output](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiYyIsInF1ZXJ5IjoidGVzdCgkJCQpIiwicmV3cml0ZSI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IFxuICAgIGNvbnRleHQ6ICRNKCQkJCk7XG4gICAgc2VsZWN0b3I6IGNhbGxfZXhwcmVzc2lvbiIsInNvdXJjZSI6IiNkZWZpbmUgdGVzdCh4KSAoMip4KVxuaW50IGEgPSB0ZXN0KDIpO1xuaW50IG1haW4oKXtcbiAgICBpbnQgYiA9IHRlc3QoMik7XG59In0=).

To avoid this ambiguity, ast-grep lets us write a [contextual pattern](/guide/rule-config/atomic-rule.html#pattern), which is a pattern inside a larger code snippet.
We can use `context` to write a pattern like this: `test($A);`. Then, we can use the selector `call_expression` to match only function calls.

### YAML

```yaml
id: match-function-call
language: c
rule:
  pattern:
    context: $M($$$);
    selector: call_expression
```

### Example

```c{2,4}
#define test(x) (2*x)
int a = test(2);
int main(){
    int b = test(2);
}
```

### Caveat

Note, tree-sitter-c parses code differently when it receives code fragment. For example,

* `test(a)` is parsed as `macro_type_specifier`
* `test(a);` is parsed as `expression_statement -> call_expression`
* `int b = test(a)` is parsed as `declaration -> init_declarator -> call_expression`

The behavior is controlled by how the tree-sitter parser is written. And tree-sitter-c behaves differently from [tree-sitter-cpp](https://github.com/tree-sitter/tree-sitter-cpp).

Please file issues on tree-sitter-c repo if you want to change the behavior. ast-grep will respect changes and decision from upstream authors.

## Rewrite Method to Function Call&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImMiLCJxdWVyeSI6IiRDT1VOVCA9ICRcbiIsInJld3JpdGUiOiIiLCJjb25maWciOiJpZDogbWV0aG9kX3JlY2VpdmVyXG5ydWxlOlxuICBwYXR0ZXJuOiAkUi4kTUVUSE9EKCQkJEFSR1MpXG50cmFuc2Zvcm06XG4gIE1BWUJFX0NPTU1BOlxuICAgIHJlcGxhY2U6XG4gICAgICBzb3VyY2U6ICQkJEFSR1NcbiAgICAgIHJlcGxhY2U6ICdeLisnXG4gICAgICBieTogJywgJ1xuZml4OlxuICAkTUVUSE9EKCYkUiRNQVlCRV9DT01NQSQkJEFSR1MpXG4iLCJzb3VyY2UiOiJ2b2lkIHRlc3RfZnVuYygpIHtcbiAgICBzb21lX3N0cnVjdC0+ZmllbGQubWV0aG9kKCk7XG4gICAgc29tZV9zdHJ1Y3QtPmZpZWxkLm90aGVyX21ldGhvZCgxLCAyLCAzKTtcbn0ifQ==)

### Description

In C, there is no built-in support for object-oriented programming, but some programmers use structs and function pointers to simulate classes and methods. However, this style can have some drawbacks, such as:

* extra memory allocation and deallocation for the struct and the function pointer.
* indirection overhead when calling the function pointer.

A possible alternative is to use a plain function call with the struct pointer as the first argument.

### YAML

```yaml
id: method_receiver
language: c
rule:
  pattern: $R.$METHOD($$$ARGS)
transform:
  MAYBE_COMMA:
    replace:
      source: $$$ARGS
      replace: '^.+'
      by: ', '
fix:
  $METHOD(&$R$MAYBE_COMMA$$$ARGS)
```

### Example

```c {2-3}
void test_func() {
    some_struct->field.method();
    some_struct->field.other_method(1, 2, 3);
}
```

### Diff

```c
void test_func() {
    some_struct->field.method(); // [!code --]
    method(&some_struct->field); // [!code ++]
    some_struct->field.other_method(1, 2, 3); // [!code --]
    other_method(&some_struct->field, 1, 2, 3); // [!code ++]
}
```

### Contributed by

[Surma](https://twitter.com/DasSurma), adapted from the [original tweet](https://twitter.com/DasSurma/status/1706086320051794217)

## Rewrite Check to Yoda Condition&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImMiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoiaWQ6IG1heS10aGUtZm9yY2UtYmUtd2l0aC15b3Vcbmxhbmd1YWdlOiBjXG5ydWxlOlxuICBwYXR0ZXJuOiAkQSA9PSAkQiBcbiAgaW5zaWRlOlxuICAgIGtpbmQ6IHBhcmVudGhlc2l6ZWRfZXhwcmVzc2lvblxuICAgIGluc2lkZToge2tpbmQ6IGlmX3N0YXRlbWVudH1cbmNvbnN0cmFpbnRzOlxuICBCOiB7IGtpbmQ6IG51bWJlcl9saXRlcmFsIH1cbmZpeDogJEIgPT0gJEEiLCJzb3VyY2UiOiJpZiAobXlOdW1iZXIgPT0gNDIpIHsgLyogLi4uICovfVxuaWYgKG5vdE1hdGNoID09IGFub3RoZXIpIHt9XG5pZiAobm90TWF0Y2gpIHt9In0=)

### Description

In programming jargon, a [Yoda condition](https://en.wikipedia.org/wiki/Yoda_conditions) is a style that places the constant portion of the expression on the left side of the conditional statement. It is used to prevent assignment errors that may occur in languages like C.

### YAML

```yaml
id: may-the-force-be-with-you
language: c
rule:
  pattern: $A == $B                 # Find equality comparison
  inside:                           # inside an if_statement
    kind: parenthesized_expression
    inside: {kind: if_statement}
constraints:                        # with the constraint that
  B: { kind: number_literal }       # right side is a number
fix: $B == $A
```

The rule targets an equality comparison, denoted by the [pattern](/guide/pattern-syntax.html) `$A == $B`. This comparison must occur [inside](/reference/rule.html#inside) an `if_statement`. Additionally, there’s a [constraint](/reference/yaml.html#constraints) that the right side of the comparison, `$B`, must be a number\_literal like `42`.

### Example

```c {1}
if (myNumber == 42) { /* ... */}
if (notMatch == another) { /* ... */}
if (notMatch) { /* ... */}
```

### Diff

```c
if (myNumber == 42) { /* ... */} // [!code --]
if (42 == myNumber) { /* ... */} // [!code ++]
if (notMatch == another) { /* ... */}
if (notMatch) { /* ... */}
```

### Contributed by

Inspired by this [thread](https://x.com/cocoa1han/status/1763020689303581141)

---

---
url: /catalog/cpp.md
---
# Cpp

This page curates a list of example ast-grep rules to check and to rewrite Cpp code.

:::tip Reuse Cpp rules with C
Cpp is a superset of C, so you can reuse Cpp rules with C code. The [`languageGlobs`](/reference/sgconfig.html#languageglobs) option can force ast-grep to parse `.c` files as Cpp.
:::

## Fix Format String Vulnerability&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImNwcCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IGZpeC1mb3JtYXQtc2VjdXJpdHktZXJyb3Jcbmxhbmd1YWdlOiBDcHBcbnJ1bGU6XG4gIHBhdHRlcm46ICRQUklOVEYoJFMsICRWQVIpXG5jb25zdHJhaW50czpcbiAgUFJJTlRGOiAjIGEgZm9ybWF0IHN0cmluZyBmdW5jdGlvblxuICAgIHsgcmVnZXg6IFwiXnNwcmludGZ8ZnByaW50ZiRcIiB9XG4gIFZBUjogIyBub3QgYSBsaXRlcmFsIHN0cmluZ1xuICAgIG5vdDpcbiAgICAgIGFueTpcbiAgICAgIC0geyBraW5kOiBzdHJpbmdfbGl0ZXJhbCB9XG4gICAgICAtIHsga2luZDogY29uY2F0ZW5hdGVkX3N0cmluZyB9XG5maXg6ICRQUklOVEYoJFMsIFwiJXNcIiwgJFZBUilcbiIsInNvdXJjZSI6Ii8vIEVycm9yXG5mcHJpbnRmKHN0ZGVyciwgb3V0KTtcbnNwcmludGYoJmJ1ZmZlclsyXSwgb2JqLT5UZXh0KTtcbnNwcmludGYoYnVmMSwgVGV4dF9TdHJpbmcoVFhUX1dBSVRJTkdfRk9SX0NPTk5FQ1RJT05TKSk7XG4vLyBPS1xuZnByaW50ZihzdGRlcnIsIFwiJXNcIiwgb3V0KTtcbnNwcmludGYoJmJ1ZmZlclsyXSwgXCIlc1wiLCBvYmotPlRleHQpO1xuc3ByaW50ZihidWYxLCBcIiVzXCIsIFRleHRfU3RyaW5nKFRYVF9XQUlUSU5HX0ZPUl9DT05ORUNUSU9OUykpOyJ9)

### Description

The [Format String exploit](https://owasp.org/www-community/attacks/Format_string_attack) occurs when the submitted data of an input string is evaluated as a command by the application.

For example, using `sprintf(s, var)` can lead to format string vulnerabilities if `var` contains user-controlled data. This can be exploited to execute arbitrary code. By explicitly specifying the format string as `"%s"`, you ensure that `var` is treated as a string, mitigating this risk.

### YAML

```yaml
id: fix-format-security-error
language: Cpp
rule:
  pattern: $PRINTF($S, $VAR)
constraints:
  PRINTF: # a format string function
    { regex: "^sprintf|fprintf$" }
  VAR: # not a literal string
    not:
      any:
      - { kind: string_literal }
      - { kind: concatenated_string }
fix: $PRINTF($S, "%s", $VAR)
```

### Example

```cpp {2-4}
// Error
fprintf(stderr, out);
sprintf(&buffer[2], obj->Text);
sprintf(buf1, Text_String(TXT_WAITING_FOR_CONNECTIONS));
// OK
fprintf(stderr, "%s", out);
sprintf(&buffer[2], "%s", obj->Text);
sprintf(buf1, "%s", Text_String(TXT_WAITING_FOR_CONNECTIONS));
```

### Diff

```js
// Error
fprintf(stderr, out); // [!code --]
fprintf(stderr, "%s", out); // [!code ++]
sprintf(&buffer[2], obj->Text); // [!code --]
sprintf(&buffer[2], "%s", obj->Text); // [!code ++]
sprintf(buf1, Text_String(TXT_WAITING_FOR_CONNECTIONS)); // [!code --]
sprintf(buf1, "%s", Text_String(TXT_WAITING_FOR_CONNECTIONS)); // [!code ++]
// OK
fprintf(stderr, "%s", out);
sprintf(&buffer[2], "%s", obj->Text);
sprintf(buf1, "%s", Text_String(TXT_WAITING_FOR_CONNECTIONS));
```

### Contributed by

[xiaoxiangmoe](https://github.com/xiaoxiangmoe)

## Find Struct Inheritance

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiY3BwIiwicXVlcnkiOiJzdHJ1Y3QgJFNPTUVUSElORzogICRJTkhFUklUU19GUk9NIHsgJCQkQk9EWTsgfSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiIsInNvdXJjZSI6InN0cnVjdCBGb286IEJhciB7fTtcblxuc3RydWN0IEJhcjogQmF6IHtcbiAgaW50IGEsIGI7XG59In0=)

### Description

ast-grep's pattern is AST based. A code snippet like `struct $SOMETHING:  $INHERITS` will not work because it does not have a correct AST structure. The correct pattern should spell out the full syntax like `struct $SOMETHING: $INHERITS { $$$BODY; }`.

Compare the ast structure below to see the difference, especially the `ERROR` node. You can also use the playground's pattern panel to debug.

:::code-group

```shell [Wrong Pattern]
ERROR
  $SOMETHING
  base_class_clause
    $INHERITS
```

```shell [Correct Pattern]
struct_specifier
  $SOMETHING
  base_class_clause
    $INHERITS
  field_declaration_list
    field_declaration
      $$$BODY
```

:::

If it is not possible to write a full pattern, [YAML rule](/guide/rule-config.html) is a better choice.

### Pattern

```shell
ast-grep --lang cpp --pattern '
struct $SOMETHING: $INHERITS { $$$BODY; }'
```

### Example

```cpp {1-3}
struct Bar: Baz {
  int a, b;
}
```

### Contributed by

Inspired by this [tweet](https://x.com/techno_bog/status/1885421768384331871)

---

---
url: /catalog/go.md
---
# Go

This page curates a list of example ast-grep rules to check and to rewrite Go code.

## Detect problematic defer statements with function calls

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiZ28iLCJxdWVyeSI6InsgXG4gICAgZGVmZXIgJEEuJEIodCwgZmFpbHBvaW50LiRNKCQkJCkpIFxufSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6ImRlZmVyX3N0YXRlbWVudCIsImNvbmZpZyI6InJ1bGU6XG4iLCJzb3VyY2UiOiJmdW5jIFRlc3RJc3N1ZTE2Njk2KHQgKnRlc3RpbmcuVCkge1xuXHRhbGFybVJhdGlvIDo9IHZhcmRlZi5NZW1vcnlVc2FnZUFsYXJtUmF0aW8uTG9hZCgpXG5cdHZhcmRlZi5NZW1vcnlVc2FnZUFsYXJtUmF0aW8uU3RvcmUoMC4wKVxuXHRkZWZlciB2YXJkZWYuTWVtb3J5VXNhZ2VBbGFybVJhdGlvLlN0b3JlKGFsYXJtUmF0aW8pXG5cdHJlcXVpcmUuTm9FcnJvcih0LCBmYWlscG9pbnQuRW5hYmxlKFwiZ2l0aHViLmNvbS9waW5nY2FwL3RpZGIvcGtnL2V4ZWN1dG9yL3NvcnRleGVjL3Rlc3RTb3J0ZWRSb3dDb250YWluZXJTcGlsbFwiLCBcInJldHVybih0cnVlKVwiKSlcblx0ZGVmZXIgcmVxdWlyZS5Ob0Vycm9yKHQsIFxuXHQgICBmYWlscG9pbnQuRGlzYWJsZShcblx0XHRcImdpdGh1Yi5jb20vcGluZ2NhcC90aWRiL3BrZy9leGVjdXRvci9zb3J0ZXhlYy90ZXN0U29ydGVkUm93Q29udGFpbmVyU3BpbGxcIlxuXHQpKVxuXHRyZXF1aXJlLk5vRXJyb3IodCwgXG5cdFx0ZmFpbHBvaW50LkVuYWJsZShcImdpdGh1Yi5jb20vcGluZ2NhcC90aWRiL3BrZy9leGVjdXRvci9qb2luL3Rlc3RSb3dDb250YWluZXJTcGlsbFwiLCBcInJldHVybih0cnVlKVwiKSlcblx0ZGVmZXIgcmVxdWlyZS5Ob0Vycm9yKHQsIFxuXHRcdGZhaWxwb2ludC5EaXNhYmxlKFwiZ2l0aHViLmNvbS9waW5nY2FwL3RpZGIvcGtnL2V4ZWN1dG9yL2pvaW4vdGVzdFJvd0NvbnRhaW5lclNwaWxsXCIpKVxufSJ9)

### Description

This rule detects a common anti-pattern in Go testing code where `defer` statements contain function calls with parameters that are evaluated immediately instead of when the defer executes.

In Go, `defer` schedules a function call to be executed when the surrounding function returns. However, the **arguments to the deferred function are evaluated immediately** when the defer statement is encountered, not when the defer executes.

This is particularly problematic when using assertion libraries in tests. For example:

```go
defer require.NoError(t, failpoint.Disable("some/path"))
```

In this case, `failpoint.Disable("some/path")` is called immediately when the defer statement is reached, not when the function exits. This means the failpoint is disabled right after being enabled, making the test ineffective.

### Pattern

```shell
ast-grep \
  --lang go \
  --pattern '{ defer $A.$B(t, failpoint.$M($$$)) } \
  --selector defer_statement'
```

### Example

```go{6-9,11-12}
func TestIssue16696(t *testing.T) {
	alarmRatio := vardef.MemoryUsageAlarmRatio.Load()
	vardef.MemoryUsageAlarmRatio.Store(0.0)
	defer vardef.MemoryUsageAlarmRatio.Store(alarmRatio)
	require.NoError(t, failpoint.Enable("github.com/pingcap/tidb/pkg/executor/sortexec/testSortedRowContainerSpill", "return(true)"))
	defer require.NoError(t,
	   failpoint.Disable(
		"github.com/pingcap/tidb/pkg/executor/sortexec/testSortedRowContainerSpill"
	))
	require.NoError(t, failpoint.Enable("github.com/pingcap/tidb/pkg/executor/join/testRowContainerSpill", "return(true)"))
	defer require.NoError(t,
		failpoint.Disable("github.com/pingcap/tidb/pkg/executor/join/testRowContainerSpill"))
}
```

### Fix

The correct way to defer a function with parameters is to wrap it in an anonymous function:

```go
defer func() {
    require.NoError(t, failpoint.Disable("some/path"))
}()
```

### Contributed by

Inspired by [YangKeao's tweet](https://x.com/YangKeao/status/1671420857565212672) about this common pitfall in TiDB codebase.

## Find function declarations with names of certain pattern

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiJyJ15bQS1aYS16MC05Xy1dKyciLCJyZXdyaXRlIjoiIiwiY29uZmlnIjoiaWQ6IHRlc3QtZnVuY3Rpb25zXG5sYW5ndWFnZTogZ29cbnJ1bGU6XG4gIGtpbmQ6IGZ1bmN0aW9uX2RlY2xhcmF0aW9uXG4gIGhhczpcbiAgICBmaWVsZDogbmFtZVxuICAgIHJlZ2V4OiBUZXN0LipcbiIsInNvdXJjZSI6InBhY2thZ2UgYWJzXG5pbXBvcnQgXCJ0ZXN0aW5nXCJcbmZ1bmMgVGVzdEFicyh0ICp0ZXN0aW5nLlQpIHtcbiAgICBnb3QgOj0gQWJzKC0xKVxuICAgIGlmIGdvdCAhPSAxIHtcbiAgICAgICAgdC5FcnJvcmYoXCJBYnMoLTEpID0gJWQ7IHdhbnQgMVwiLCBnb3QpXG4gICAgfVxufVxuIn0=)

### Description

ast-grep can find function declarations by their names. But not all names can be matched by a meta variable pattern. For instance, you cannot use a meta variable pattern to find function declarations whose names start with a specific prefix, e.g. `TestAbs` with the prefix `Test`. Attempting `Test$_` will fail because it is not a valid syntax.

Instead, you can use a [YAML rule](/reference/rule.html) to use the [`regex`](/guide/rule-config/atomic-rule.html#regex) atomic rule.

### YAML

```yaml
id: test-functions
language: go
rule:
  kind: function_declaration
  has:
    field: name
    regex: Test.*
```

### Example

```go{3-8}
package abs
import "testing"
func TestAbs(t *testing.T) {
    got := Abs(-1)
    if got != 1 {
        t.Errorf("Abs(-1) = %d; want 1", got)
    }
}
```

### Contributed by

[kevinkjt2000](https://twitter.com/kevinkjt2000) on [Discord](https://discord.com/invite/4YZjf6htSQ).

## Match Function Call in Golang

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiJhd2FpdCAkQSIsInJld3JpdGUiOiJ0cnkge1xuICAgIGF3YWl0ICRBXG59IGNhdGNoKGUpIHtcbiAgICAvLyB0b2RvXG59IiwiY29uZmlnIjoicnVsZTpcbiAgcGF0dGVybjpcbiAgICBjb250ZXh0OiAnZnVuYyB0KCkgeyBmbXQuUHJpbnRsbigkJCRBKSB9J1xuICAgIHNlbGVjdG9yOiBjYWxsX2V4cHJlc3Npb25cbiIsInNvdXJjZSI6ImZ1bmMgbWFpbigpIHtcbiAgICBmbXQuUHJpbnRsbihcIk9LXCIpXG59In0=)

### Description

One of the common questions of ast-grep is to match function calls in Golang.

A plain pattern like `fmt.Println($A)` will not work. This is because Golang syntax also allows type conversions, e.g. `int(3.14)`, that look like function calls. Tree-sitter, ast-grep's parser, will prefer parsing `func_call(arg)` as a type conversion instead of a call expression.

To avoid this ambiguity, ast-grep lets us write a [contextual pattern](/guide/rule-config/atomic-rule.html#pattern), which is a pattern inside a larger code snippet.
We can use `context` to write a pattern like this: `func t() { fmt.Println($A) }`. Then, we can use the selector `call_expression` to match only function calls.

Please also read the [deep dive](/advanced/pattern-parse.html) on [ambiguous pattern](/advanced/pattern-parse.html#ambiguous-pattern-code).

### YAML

```yaml
id: match-function-call
language: go
rule:
  pattern:
    context: 'func t() { fmt.Println($A) }'
    selector: call_expression
```

### Example

```go{2}
func main() {
    fmt.Println("OK")
}
```

### Contributed by

Inspired by [QuantumGhost](https://github.com/QuantumGhost) from [ast-grep/ast-grep#646](https://github.com/ast-grep/ast-grep/issues/646)

## Match package import in Golang

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiIiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogbWF0Y2gtcGFja2FnZS1pbXBvcnRcbmxhbmd1YWdlOiBnb1xucnVsZTpcbiAga2luZDogaW1wb3J0X3NwZWNcbiAgaGFzOlxuICAgIHJlZ2V4OiBnaXRodWIuY29tL2dvbGFuZy1qd3Qvand0Iiwic291cmNlIjoicGFja2FnZSBtYWluXG5cbmltcG9ydCAoXG5cdFwiZm10XCJcblx0XCJnaXRodWIuY29tL2dvbGFuZy1qd3Qvand0XCIgIC8vIFRoaXMgbWF0Y2hlcyB0aGUgQVNUIHJ1bGVcbilcblxuZnVuYyBtYWluKCkge1xuXHQvLyBDcmVhdGUgYSBuZXcgdG9rZW5cblx0dG9rZW4gOj0gand0Lk5ldyhqd3QuU2lnbmluZ01ldGhvZEhTMjU2KVxuXHRcblx0Ly8gQWRkIHNvbWUgY2xhaW1zXG5cdHRva2VuLkNsYWltcyA9IGp3dC5NYXBDbGFpbXN7XG5cdFx0XCJ1c2VyXCI6IFwiYWxpY2VcIixcblx0XHRcInJvbGVcIjogXCJhZG1pblwiLFxuXHR9XG5cdFxuXHQvLyBTaWduIHRoZSB0b2tlblxuXHR0b2tlblN0cmluZywgZXJyIDo9IHRva2VuLlNpZ25lZFN0cmluZyhbXWJ5dGUoXCJteS1zZWNyZXRcIikpXG5cdGlmIGVyciAhPSBuaWwge1xuXHRcdGZtdC5QcmludGYoXCJFcnJvciBzaWduaW5nIHRva2VuOiAldlxcblwiLCBlcnIpXG5cdFx0cmV0dXJuXG5cdH1cblx0XG5cdGZtdC5QcmludGYoXCJHZW5lcmF0ZWQgdG9rZW46ICVzXFxuXCIsIHRva2VuU3RyaW5nKVxufSJ9)

### Description

A generic rule template for detecting imports of specific packages in Go source code. This rule can be customized to match any package by modifying the regex pattern, making it useful for security auditing, dependency management, and compliance checking.

This rule identifies Go import statements based on the configured regex pattern, including:

Direct imports: `import "package/name"`\
Versioned imports: `import "package/name/v4"`\
Subpackage imports: `import "package/name/subpkg"`\
Grouped imports within `import () blocks`

### YAML

```yaml
id: match-package-import
language: go
rule:
  kind: import_spec
  has:
    regex: PACKAGE_PATTERN_HERE
```

### Example

JWT Library Detection

```go{5}
package main

import (
	"fmt"
	"github.com/golang-jwt/jwt" // This matches the AST rule
)

func main() {
	token := jwt.New(jwt.SigningMethodHS256) // Create a new token
	// Add some claims
	token.Claims = jwt.MapClaims{"user": "alice", "role": "admin"}
	tokenString, err := token.SignedString([]byte("my-secret")) // Sign the token
	if err != nil {
		fmt.Printf("Error signing token: %v\n", err)
		return
	}
	fmt.Printf("Generated token: %s\n", tokenString)
}
```

### Contributed by

[Sudesh Gutta](https://github.com/sudeshgutta)

## Detect problematic JSON tags with dash prefix

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiJgJFRBR2AiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogdW5tYXJzaGFsLXRhZy1pcy1kYXNoXG5zZXZlcml0eTogZXJyb3Jcbm1lc3NhZ2U6IFN0cnVjdCBmaWVsZCBjYW4gYmUgZGVjb2RlZCB3aXRoIHRoZSBgLWAga2V5IGJlY2F1c2UgdGhlIEpTT04gdGFnXG4gIHN0YXJ0cyB3aXRoIGEgYC1gIGJ1dCBpcyBmb2xsb3dlZCBieSBhIGNvbW1hLlxucnVsZTpcbiAgcGF0dGVybjogJ2AkVEFHYCdcbiAgaW5zaWRlOlxuICAgIGtpbmQ6IGZpZWxkX2RlY2xhcmF0aW9uXG5jb25zdHJhaW50czpcbiAgVEFHOiBcbiAgICByZWdleDoganNvbjpcIi0sLipcIiIsInNvdXJjZSI6InBhY2thZ2UgbWFpblxuXG50eXBlIFRlc3RTdHJ1Y3QxIHN0cnVjdCB7XG5cdC8vIG9rOiB1bm1hcnNoYWwtdGFnLWlzLWRhc2hcblx0QSBzdHJpbmcgYGpzb246XCJpZFwiYFxufVxuXG50eXBlIFRlc3RTdHJ1Y3QyIHN0cnVjdCB7XG5cdC8vIHJ1bGVpZDogdW5tYXJzaGFsLXRhZy1pcy1kYXNoXG5cdEIgc3RyaW5nIGBqc29uOlwiLSxvbWl0ZW1wdHlcImBcbn1cblxudHlwZSBUZXN0U3RydWN0MyBzdHJ1Y3Qge1xuXHQvLyBydWxlaWQ6IHVubWFyc2hhbC10YWctaXMtZGFzaFxuXHRDIHN0cmluZyBganNvbjpcIi0sMTIzXCJgXG59XG5cbnR5cGUgVGVzdFN0cnVjdDQgc3RydWN0IHtcblx0Ly8gcnVsZWlkOiB1bm1hcnNoYWwtdGFnLWlzLWRhc2hcblx0RCBzdHJpbmcgYGpzb246XCItLFwiYFxufSJ9)

### Description

This rule detects a security vulnerability in Go's JSON unmarshaling. When a struct field has a JSON tag that starts with `-,`, it can be unexpectedly unmarshaled with the `-` key.

According to the [Go documentation](https://pkg.go.dev/encoding/json#Marshal), if the field tag is `-`, the field should be omitted. However, a field with name `-` can still be unmarshaled using the tag `-,`.

This creates a security issue where developers think they are preventing a field from being unmarshaled (like `IsAdmin` in authentication), but attackers can still set that field by providing the `-` key in JSON input.

```go
type User struct {
    Username string `json:"username,omitempty"`
    Password string `json:"password,omitempty"`
    IsAdmin  bool   `json:"-,omitempty"`  // Intended to prevent marshaling
}

// This still works and sets IsAdmin to true!
json.Unmarshal([]byte(`{"-": true}`), &user)
// Result: main.User{Username:"", Password:"", IsAdmin:true}
```

### YAML

```yaml
id: unmarshal-tag-is-dash
severity: error
message: Struct field can be decoded with the `-` key because the JSON tag
  starts with a `-` but is followed by a comma.
rule:
  pattern: '`$TAG`'
  inside:
    kind: field_declaration
constraints:
  TAG:
    regex: json:"-,.*"
```

### Example

```go{8,12,16}
package main

type TestStruct1 struct {
	A string `json:"id"` // ok
}

type TestStruct2 struct {
	B string `json:"-,omitempty"` // wrong
}

type TestStruct3 struct {
	C string `json:"-,123"` // wrong
}

type TestStruct4 struct {
	D string `json:"-,"` // wrong
}
```

### Fix

To properly omit a field from JSON marshaling/unmarshaling, use just `-` without a comma:

```go
type User struct {
    Username string `json:"username,omitempty"`
    Password string `json:"password,omitempty"`
    IsAdmin  bool   `json:"-"`  // Correctly prevents marshaling/unmarshaling
}
```

### Contributed by

Inspired by [Trail of Bits blog post](https://blog.trailofbits.com/2025/06/17/unexpected-security-footguns-in-gos-parsers/) and their [public Semgrep rule](https://semgrep.dev/playground/r/trailofbits.go.unmarshal-tag-is-dash).

---

---
url: /catalog/html.md
---
# HTML

This page curates a list of example ast-grep rules to check and to rewrite HTML code.

:::tip Use HTML parser for frameworks
You can leverage the [`languageGlobs`](/reference/sgconfig.html#languageglobs) option to parse framework files as plain HTML, such as `vue`, `svelte`, and `astro`.

**Caveat**: This approach may not parse framework-specific syntax, like Astro's [frontmatter script](https://docs.astro.build/en/basics/astro-components/#the-component-script) or [Svelte control flow](https://svelte.dev/docs/svelte/if). You will need to load [custom languages](/advanced/custom-language.html) for such cases.
:::

## Upgrade Ant Design Vue&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6Imh0bWwiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoidXRpbHM6XG4gIGluc2lkZS10YWc6XG4gICAgaW5zaWRlOlxuICAgICAga2luZDogZWxlbWVudCBcbiAgICAgIHN0b3BCeTogeyBraW5kOiBlbGVtZW50IH1cbiAgICAgIGhhczpcbiAgICAgICAgc3RvcEJ5OiB7IGtpbmQ6IHRhZ19uYW1lIH1cbiAgICAgICAga2luZDogdGFnX25hbWVcbiAgICAgICAgcGF0dGVybjogJFRBR19OQU1FXG5ydWxlOlxuICBraW5kOiBhdHRyaWJ1dGVfbmFtZVxuICByZWdleDogOnZpc2libGVcbiAgbWF0Y2hlczogaW5zaWRlLXRhZyAgXG5maXg6IDpvcGVuXG5jb25zdHJhaW50czpcbiAgVEFHX05BTUU6XG4gICAgcmVnZXg6IGEtbW9kYWx8YS10b29sdGlwIiwic291cmNlIjoiPHRlbXBsYXRlPlxuICA8YS1tb2RhbCA6dmlzaWJsZT1cInZpc2libGVcIj5jb250ZW50PC9hLW1vZGFsPlxuICA8YS10b29sdGlwIDp2aXNpYmxlPVwidmlzaWJsZVwiIC8+XG4gIDxhLXRhZyA6dmlzaWJsZT1cInZpc2libGVcIj50YWc8L2EtdGFnPlxuPC90ZW1wbGF0ZT4ifQ==)

### Description

ast-grep can be used to upgrade Vue template using the HTML parser.

This rule is an example to upgrade [one breaking change](https://next.antdv.com/docs/vue/migration-v4#component-api-adjustment) in [Ant Design Vue](https://next.antdv.com/components/overview) from v3 to v4, unifying the controlled visible API of the component popup.

It is designed to identify and replace the `visible` attribute with the `open` attribute for specific components like `a-modal` and `a-tooltip`. Note the rule should not replace other visible attributes that are not related to the component popup like `a-tag`.

The rule can be broken down into the following steps:

1. Find the target attribute name by `kind` and `regex`
2. Find the attribute's enclosing element using `inside`, and get its tag name
3. Ensure the tag name is related to popup components, using constraints

### YAML

```yaml
id: upgrade-ant-design-vue
language: HTML
utils:
  inside-tag:
    # find the enclosing element of the attribute
    inside:
      kind: element
      stopBy: { kind: element } # only the closest element
      # find the tag name and store it in metavar
      has:
        stopBy: { kind: tag_name }
        kind: tag_name
        pattern: $TAG_NAME
rule:
  # find the target attribute_name
  kind: attribute_name
  regex: :visible
  # find the element
  matches: inside-tag
# ensure it only matches modal/tooltip but not tag
constraints:
  TAG_NAME:
    regex: a-modal|a-tooltip
fix: :open
```

### Example

```html {2,3}
<template>
  <a-modal :visible="visible">content</a-modal>
  <a-tooltip :visible="visible" />
  <a-tag :visible="visible">tag</a-tag>
</template>
```

### Diff

```html
<template>
  <a-modal :visible="visible">content</a-modal> // [!code --]
  <a-modal :open="visible">content</a-modal> // [!code ++]
  <a-tooltip :visible="visible" /> // [!code --]
  <a-tooltip :open="visible" /> // [!code ++]
  <a-tag :visible="visible">tag</a-tag>
</template>
```

### Contributed by

Inspired by [Vue.js RFC](https://github.com/vuejs/rfcs/discussions/705#discussion-7255672)

## Extract i18n Keys&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6Imh0bWwiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogdGV4dFxuICBwYXR0ZXJuOiAkVFxuICBub3Q6XG4gICAgcmVnZXg6ICdcXHtcXHsuKlxcfVxcfSdcbmZpeDogXCJ7eyAkKCckVCcpIH19XCIiLCJzb3VyY2UiOiI8dGVtcGxhdGU+XG4gIDxzcGFuPkhlbGxvPC9zcGFuPlxuICA8c3Bhbj57eyB0ZXh0IH19PC9zcGFuPlxuPC90ZW1wbGF0ZT4ifQ==)

### Description

It is tedious to manually find and replace all the text in the template with i18n keys. This rule helps to extract static text into i18n keys. Dynamic text, e.g. mustache syntax, will be skipped.

In practice, you may want to map the extracted text to a key in a dictionary file. While this rule only demonstrates the extraction part, further mapping process can be done via a script reading the output of ast-grep's [`--json`](/guide/tools/json.html) mode, or using [`@ast-grep/napi`](/guide/api-usage/js-api.html).

### YAML

```yaml
id: extract-i18n-key
language: html
rule:
  kind: text
  pattern: $T
  # skip dynamic text in mustache syntax
  not: { regex: '\{\{.*\}\}' }
fix: "{{ $('$T') }}"
```

### Example

```html {2}
<template>
  <span>Hello</span>
  <span>{{ text }}</span>
</template>
```

### Diff

```html
<template>
  <span>Hello</span> // [!code --]
  <span>{{ $('Hello') }}</span> // [!code ++]
  <span>{{ text }}</span>
</template>
```

### Contributed by

Inspired by [Vue.js RFC](https://github.com/vuejs/rfcs/discussions/705#discussion-7255672)

---

---
url: /catalog/java.md
---
# Java

This page curates a list of example ast-grep rules to check and to rewrite Java code.

## No Unused Vars in Java&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmEiLCJxdWVyeSI6ImlmKHRydWUpeyQkJEJPRFl9IiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogbm8tdW51c2VkLXZhcnNcbnJ1bGU6XG4gICAga2luZDogbG9jYWxfdmFyaWFibGVfZGVjbGFyYXRpb25cbiAgICBhbGw6XG4gICAgICAgIC0gaGFzOlxuICAgICAgICAgICAgaGFzOlxuICAgICAgICAgICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgICAgICAgICBwYXR0ZXJuOiAkSURFTlRcbiAgICAgICAgLSBub3Q6XG4gICAgICAgICAgICBwcmVjZWRlczpcbiAgICAgICAgICAgICAgICBzdG9wQnk6IGVuZFxuICAgICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAgICAgICAgICAgICAgICAgYW55OlxuICAgICAgICAgICAgICAgICAgICAgICAgLSB7IGtpbmQ6IGlkZW50aWZpZXIsIHBhdHRlcm46ICRJREVOVCB9XG4gICAgICAgICAgICAgICAgICAgICAgICAtIHsgaGFzOiB7a2luZDogaWRlbnRpZmllciwgcGF0dGVybjogJElERU5ULCBzdG9wQnk6IGVuZH19XG5maXg6ICcnXG4iLCJzb3VyY2UiOiJTdHJpbmcgdW51c2VkID0gXCJ1bnVzZWRcIjtcbk1hcDxTdHJpbmcsIFN0cmluZz4gZGVjbGFyZWRCdXROb3RJbnN0YW50aWF0ZWQ7XG5cblN0cmluZyB1c2VkMSA9IFwidXNlZFwiO1xuaW50IHVzZWQyID0gMztcbmJvb2xlYW4gdXNlZDMgPSBmYWxzZTtcbmludCB1c2VkNCA9IDQ7XG5TdHJpbmcgdXNlZDUgPSBcIjVcIjtcblxuXG5cbnVzZWQxO1xuU3lzdGVtLm91dC5wcmludGxuKHVzZWQyKTtcbmlmKHVzZWQzKXtcbiAgICBTeXN0ZW0ub3V0LnByaW50bG4oXCJzb21lIHZhcnMgYXJlIHVudXNlZFwiKTtcbiAgICBNYXA8U3RyaW5nLCBTdHJpbmc+IHVudXNlZE1hcCA9IG5ldyBIYXNoTWFwPD4oKSB7e1xuICAgICAgICBwdXQodXNlZDUsIFwidXNlZDVcIik7XG4gICAgfX07XG5cbiAgICAvLyBFdmVuIHRob3VnaCB3ZSBkb24ndCByZWFsbHkgZG8gYW55dGhpbmcgd2l0aCB0aGlzIG1hcCwgc2VwYXJhdGluZyB0aGUgZGVjbGFyYXRpb24gYW5kIGluc3RhbnRpYXRpb24gbWFrZXMgaXQgY291bnQgYXMgYmVpbmcgdXNlZFxuICAgIGRlY2xhcmVkQnV0Tm90SW5zdGFudGlhdGVkID0gbmV3IEhhc2hNYXA8PigpO1xuXG4gICAgcmV0dXJuIHVzZWQ0O1xufSJ9)

### Description

Identifying unused variables is a common task in code refactoring. You should rely on a Java linter or IDE for this task rather than writing a custom rule in ast-grep, but for educational purposes, this rule demonstrates how to find unused variables in Java.

This approach makes some simplifying assumptions. We only consider local variable declarations and ignore the other many ways variables can be declared: Method Parameters, Fields, Class Variables, Constructor Parameters, Loop Variables, Exception Handler Parameters, Lambda Parameters, Annotation Parameters, Enum Constants, and Record Components. Now you may see why it is recommended to use a rule from an established linter or IDE rather than writing your own.

### YAML

```yaml
id: no-unused-vars
rule:
    kind: local_variable_declaration
    all:
        - has:
            has:
                kind: identifier
                pattern: $IDENT
        - not:
            precedes:
                stopBy: end
                has:
                    stopBy: end
                    any:
                        - { kind: identifier, pattern: $IDENT }
                        - { has: {kind: identifier, pattern: $IDENT, stopBy: end}}
fix: ''
```

First, we identify the local variable declaration and capture the pattern of the identifier inside of it. Then we use `not` and `precedes` to only match the local variable declaration if the identifier we captured does not appear later in the code.

It is important to note that we use `all` here to force the ordering of the `has` rule to be before the `not` rule. This guarantees that the meta-variable `$IDENT` is captured by looking inside of the local variable declaration.

Additionally, when looking ahead in the code, we can't just look for the identifier directly, but for any node that may contain the identifier.

### Example

```java
String unused = "unused"; // [!code --]
String used = "used";
System.out.println(used);
```

## Find Java field declarations of type String

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmEiLCJxdWVyeSI6ImAkVEFHYCIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIGtpbmQ6IGZpZWxkX2RlY2xhcmF0aW9uXG4gIGhhczpcbiAgICBmaWVsZDogdHlwZVxuICAgIHJlZ2V4OiBeU3RyaW5nJCIsInNvdXJjZSI6IkBDb21wb25lbnRcbmNsYXNzIEFCQyBleHRlbmRzIE9iamVjdHtcbiAgICBAUmVzb3VyY2VcbiAgICBwcml2YXRlIGZpbmFsIFN0cmluZyB3aXRoX2Fubm87XG5cbiAgICBwcml2YXRlIGZpbmFsIFN0cmluZyB3aXRoX211bHRpX21vZDtcblxuICAgIHB1YmxpYyBTdHJpbmcgc2ltcGxlO1xufSJ9)

### Description

To extract all Java field names of type `String` is not as straightforward as one might think. A simple pattern like `String $F;` would only match fields declared without any modifiers or annotations. However, a pattern like `$MOD String $F;` cannot be correctly parsed by tree-sitter.

:::details Use playground pattern debugger to explore the AST

You can use the [playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YSIsInF1ZXJ5IjoiY2xhc3MgQUJDe1xuICAgJE1PRCBTdHJpbmcgdGVzdDtcbn0iLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6ImFzdCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogZmllbGRfZGVjbGFyYXRpb25cbiAgaGFzOlxuICAgIGZpZWxkOiB0eXBlXG4gICAgcmVnZXg6IF5TdHJpbmckIiwic291cmNlIjoiQENvbXBvbmVudFxuY2xhc3MgQUJDIGV4dGVuZHMgT2JqZWN0e1xuICAgIEBSZXNvdXJjZVxuICAgIHByaXZhdGUgZmluYWwgU3RyaW5nIHdpdGhfYW5ubztcblxuICAgIHByaXZhdGUgZmluYWwgU3RyaW5nIHdpdGhfbXVsdGlfbW9kO1xuXG4gICAgcHVibGljIFN0cmluZyBzaW1wbGU7XG59In0=)'s pattern tab to visualize the AST of `class A { $MOD String $F; }`.

```
field_declaration
  $MOD
  variable_declarator
    identifier: String
  ERROR
    identifier: $F
```

Tree-sitter does not think `$MOD` is a valid modifier, so it produces an `ERROR`.

While the valid AST for code like `private String field;` produces different AST structures:

```
field_declaration
  modifiers
  type_identifier
  variable_declarator
    identifier: field
```

:::

A more robust approach is to use a structural rule that targets `field_declaration` nodes and applies a `has` constraint on the `type` child node to match the type `String`. This method effectively captures fields regardless of their modifiers or annotations.

### YAML

```yaml
id: find-field-with-type
language: java
rule:
  kind: field_declaration
  has:
    field: type
    regex: ^String$
```

### Example

```java {3-4,6,8}
@Component
class ABC extends Object{
    @Resource
    private final String with_anno;

    private final String with_multi_mod;

    public String simple;
}
```

### Contributed by

Inspired by the post [discussion](https://github.com/ast-grep/ast-grep/discussions/2195)

---

---
url: /catalog/kotlin.md
---
# Kotlin

This page curates a list of example ast-grep rules to check and to rewrite Kotlin code.

## Ensure Clean Architecture

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImtvdGxpbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogaW1wb3J0LWRlcGVuZGVuY3ktdmlvbGF0aW9uXG5tZXNzYWdlOiBJbXBvcnQgRGVwZW5kZW5jeSBWaW9sYXRpb24gXG5ub3RlczogRW5zdXJlcyB0aGF0IGltcG9ydHMgY29tcGx5IHdpdGggYXJjaGl0ZWN0dXJhbCBydWxlcy4gXG5zZXZlcml0eTogZXJyb3JcbnJ1bGU6XG4gIHBhdHRlcm46IGltcG9ydCAkUEFUSFxuY29uc3RyYWludHM6XG4gIFBBVEg6XG4gICAgYW55OlxuICAgIC0gcmVnZXg6IGNvbVxcLmV4YW1wbGUoXFwuXFx3KykqXFwuZGF0YVxuICAgIC0gcmVnZXg6IGNvbVxcLmV4YW1wbGUoXFwuXFx3KykqXFwucHJlc2VudGF0aW9uXG5maWxlczpcbi0gY29tL2V4YW1wbGUvZG9tYWluLyoqLyoua3QiLCJzb3VyY2UiOiJpbXBvcnQgYW5kcm9pZHgubGlmZWN5Y2xlLlZpZXdNb2RlbFxuaW1wb3J0IGFuZHJvaWR4LmxpZmVjeWNsZS5WaWV3TW9kZWxTY29wZVxuaW1wb3J0IGNvbS5leGFtcGxlLmN1c3RvbWxpbnRleGFtcGxlLmRhdGEubW9kZWxzLlVzZXJEdG9cbmltcG9ydCBjb20uZXhhbXBsZS5jdXN0b21saW50ZXhhbXBsZS5kb21haW4udXNlY2FzZXMuR2V0VXNlclVzZUNhc2VcbmltcG9ydCBjb20uZXhhbXBsZS5jdXN0b21saW50ZXhhbXBsZS5wcmVzZW50YXRpb24uc3RhdGVzLk1haW5TdGF0ZVxuaW1wb3J0IGRhZ2dlci5oaWx0LmFuZHJvaWQubGlmZWN5Y2xlLkhpbHRWaWV3TW9kZWwifQ==)

### Description

This ast-grep rule ensures that the **domain** package in a [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) project does not import classes from the **data** or **presentation** packages. It enforces the separation of concerns by preventing the domain layer from depending on other layers, maintaining the integrity of the architecture.

For example, the rule will trigger an error if an import statement like `import com.example.data.SomeClass` or `import com.example.presentation.AnotherClass` is found within the domain package.

The rule uses the [`files`](/reference/yaml.html#files) field to apply only to the domain package.

### YAML

```yaml
id: import-dependency-violation
message: Import Dependency Violation
notes: Ensures that imports comply with architectural rules.
severity: error
rule:
  pattern: import $PATH  # capture the import statement
constraints:
  PATH: # find specific package imports
    any:
    - regex: com\.example(\.\w+)*\.data
    - regex: com\.example(\.\w+)*\.presentation
files:  # apply only to domain package
- com/example/domain/**/*.kt
```

### Example

```kotlin {3,5}
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelScope
import com.example.customlintexample.data.models.UserDto
import com.example.customlintexample.domain.usecases.GetUserUseCase
import com.example.customlintexample.presentation.states.MainState
import dagger.hilt.android.lifecycle.HiltViewModel
```

### Contributed by

Inspired by the post [Custom Lint Task Configuration in Gradle with Kotlin DSL](https://www.sngular.com/insights/320/custom-lint-task-configuration-in-gradle-with-kotlin-dsl)

---

---
url: /catalog/python.md
---
# Python

This page curates a list of example ast-grep rules to check and to rewrite Python code.

## Migrate OpenAI SDK&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmICRGVU5DKCQkJEFSR1MpOiAkJCRCT0RZIiwicmV3cml0ZSI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IGltcG9ydCBvcGVuYWlcbmZpeDogZnJvbSBvcGVuYWkgaW1wb3J0IENsaWVudFxuLS0tXG5ydWxlOlxuICBwYXR0ZXJuOiBvcGVuYWkuYXBpX2tleSA9ICRLRVlcbmZpeDogY2xpZW50ID0gQ2xpZW50KCRLRVkpXG4tLS1cbnJ1bGU6XG4gIHBhdHRlcm46IG9wZW5haS5Db21wbGV0aW9uLmNyZWF0ZSgkJCRBUkdTKVxuZml4OiB8LVxuICBjbGllbnQuY29tcGxldGlvbnMuY3JlYXRlKFxuICAgICQkJEFSR1NcbiAgKSIsInNvdXJjZSI6ImltcG9ydCBvc1xuaW1wb3J0IG9wZW5haVxuZnJvbSBmbGFzayBpbXBvcnQgRmxhc2ssIGpzb25pZnlcblxuYXBwID0gRmxhc2soX19uYW1lX18pXG5vcGVuYWkuYXBpX2tleSA9IG9zLmdldGVudihcIk9QRU5BSV9BUElfS0VZXCIpXG5cblxuQGFwcC5yb3V0ZShcIi9jaGF0XCIsIG1ldGhvZHM9KFwiUE9TVFwiKSlcbmRlZiBpbmRleCgpOlxuICAgIGFuaW1hbCA9IHJlcXVlc3QuZm9ybVtcImFuaW1hbFwiXVxuICAgIHJlc3BvbnNlID0gb3BlbmFpLkNvbXBsZXRpb24uY3JlYXRlKFxuICAgICAgICBtb2RlbD1cInRleHQtZGF2aW5jaS0wMDNcIixcbiAgICAgICAgcHJvbXB0PWdlbmVyYXRlX3Byb21wdChhbmltYWwpLFxuICAgICAgICB0ZW1wZXJhdHVyZT0wLjYsXG4gICAgKVxuICAgIHJldHVybiBqc29uaWZ5KHJlc3BvbnNlLmNob2ljZXMpIn0=)

### Description

OpenAI has introduced some breaking changes in their API, such as using `Client` to initialize the service and renaming the `Completion` method to `completions` . This example shows how to use ast-grep to automatically update your code to the new API.

API migration requires multiple related rules to work together.
The example shows how to write [multiple rules](/reference/playground.html#test-multiple-rules) in a [single YAML](/guide/rewrite-code.html#using-fix-in-yaml-rule) file.
The rules and patterns in the example are simple and self-explanatory, so we will not explain them further.

### YAML

```yaml
id: import-openai
language: python
rule:
  pattern: import openai
fix: from openai import Client
---
id: rewrite-client
language: python
rule:
  pattern: openai.api_key = $KEY
fix: client = Client($KEY)
---
id: rewrite-chat-completion
language: python
rule:
  pattern: openai.Completion.create($$$ARGS)
fix: |-
  client.completions.create(
    $$$ARGS
  )
```

### Example

```python {2,6,11-15}
import os
import openai
from flask import Flask, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/chat", methods=("POST"))
def index():
    animal = request.form["animal"]
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(animal),
        temperature=0.6,
    )
    return jsonify(response.choices)
```

### Diff

```python
import os
import openai # [!code --]
from openai import Client # [!code ++]
from flask import Flask, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY") # [!code --]
client = Client(os.getenv("OPENAI_API_KEY")) # [!code ++]

@app.route("/chat", methods=("POST"))
def index():
    animal = request.form["animal"]
    response = openai.Completion.create( # [!code --]
    response = client.completions.create( # [!code ++]
      model="text-davinci-003",
      prompt=generate_prompt(animal),
      temperature=0.6,
    )
    return jsonify(response.choices)
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [Morgante](https://twitter.com/morgantepell/status/1721668781246750952) from [grit.io](https://www.grit.io/)

## Prefer Generator Expressions&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiWyQkJEFdIiwicmV3cml0ZSI6IiRBPy4oKSIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46ICRGVU5DKCRMSVNUKVxuY29uc3RyYWludHM6XG4gIExJU1Q6IHsga2luZDogbGlzdF9jb21wcmVoZW5zaW9uIH1cbiAgRlVOQzpcbiAgICBhbnk6XG4gICAgICAtIHBhdHRlcm46IGFueVxuICAgICAgLSBwYXR0ZXJuOiBhbGxcbiAgICAgIC0gcGF0dGVybjogc3VtXG4gICAgICAjIC4uLlxudHJhbnNmb3JtOlxuICBJTk5FUjpcbiAgICBzdWJzdHJpbmc6IHtzb3VyY2U6ICRMSVNULCBzdGFydENoYXI6IDEsIGVuZENoYXI6IC0xIH1cbmZpeDogJEZVTkMoJElOTkVSKSIsInNvdXJjZSI6ImFsbChbeCBmb3IgeCBpbiB5XSlcblt4IGZvciB4IGluIHldIn0=)

### Description

List comprehensions like `[x for x in range(10)]` are a concise way to create lists in Python. However, we can achieve better memory efficiency by using generator expressions like `(x for x in range(10))` instead. List comprehensions create the entire list in memory, while generator expressions generate each element one at a time. We can make the change by replacing the square brackets with parentheses.

### YAML

```yaml
id: prefer-generator-expressions
language: python
rule:
  pattern: $LIST
  kind: list_comprehension
transform:
  INNER:
    substring: {source: $LIST, startChar: 1, endChar: -1 }
fix: ($INNER)
```

This rule converts every list comprehension to a generator expression. However, **not every list comprehension can be replaced with a generator expression.** If the list is used multiple times, is modified, is sliced, or is indexed, a generator is not a suitable replacement.

Some common functions like `any`, `all`, and `sum` take an `iterable` as an argument. A generator function counts as an `iterable`, so it is safe to change a list comprehension to a generator expression in this context.

```yaml
id: prefer-generator-expressions
language: python
rule:
  pattern: $FUNC($LIST)
constraints:
  LIST: { kind: list_comprehension }
  FUNC:
    any:
      - pattern: any
      - pattern: all
      - pattern: sum
      # ...
transform:
  INNER:
    substring: {source: $LIST, startChar: 1, endChar: -1 }
fix: $FUNC($INNER)
```

### Example

```python
any([x for x in range(10)])
```

### Diff

```python
any([x for x in range(10)]) # [!code --]
any(x for x in range(10)) # [!code ++]
```

### Contributed by

[Steven Love](https://github.com/StevenLove)

## Use Walrus Operator in `if` statement

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZm4gbWFpbigpIHsgXG4gICAgJCQkO1xuICAgIGlmKCRBKXskJCRCfSBcbiAgICBpZigkQSl7JCQkQ30gXG4gICAgJCQkRlxufSIsInJld3JpdGUiOiJmbiBtYWluKCkgeyAkJCRFOyBpZigkQSl7JCQkQiAkJCRDfSAkJCRGfSIsImNvbmZpZyI6ImlkOiB1c2Utd2FscnVzLW9wZXJhdG9yXG5ydWxlOlxuICBmb2xsb3dzOlxuICAgIHBhdHRlcm46XG4gICAgICBjb250ZXh0OiAkVkFSID0gJCQkRVhQUlxuICAgICAgc2VsZWN0b3I6IGV4cHJlc3Npb25fc3RhdGVtZW50XG4gIHBhdHRlcm46IFwiaWYgJFZBUjogJCQkQlwiXG5maXg6IHwtXG4gIGlmICRWQVIgOj0gJCQkRVhQUjpcbiAgICAkJCRCXG4tLS1cbmlkOiByZW1vdmUtZGVjbGFyYXRpb25cbnJ1bGU6XG4gIHBhdHRlcm46XG4gICAgY29udGV4dDogJFZBUiA9ICQkJEVYUFJcbiAgICBzZWxlY3RvcjogZXhwcmVzc2lvbl9zdGF0ZW1lbnRcbiAgcHJlY2VkZXM6XG4gICAgcGF0dGVybjogXCJpZiAkVkFSOiAkJCRCXCJcbmZpeDogJyciLCJzb3VyY2UiOiJhID0gZm9vKClcblxuaWYgYTpcbiAgICBkb19iYXIoKSJ9)

### Description

The walrus operator (`:=`) introduced in Python 3.8 allows you to assign values to variables as part of an expression. This rule aims to simplify code by using the walrus operator in `if` statements.

This first part of the rule identifies cases where a variable is assigned a value and then immediately used in an `if` statement to control flow.

```yaml
id: use-walrus-operator
language: python
rule:
  pattern: "if $VAR: $$$B"
  follows:
    pattern:
      context: $VAR = $$$EXPR
      selector: expression_statement
fix: |-
  if $VAR := $$$EXPR:
    $$$B
```

The `pattern` clause finds an `if` statement that checks the truthiness of `$VAR`.
If this pattern `follows` an expression statement where `$VAR` is assigned `$$$EXPR`, the `fix` clause changes the `if` statements to use the walrus operator.

The second part of the rule:

```yaml
id: remove-declaration
rule:
  pattern:
    context: $VAR = $$$EXPR
    selector: expression_statement
  precedes:
    pattern: "if $VAR: $$$B"
fix: ''
```

This rule removes the standalone variable assignment when it directly precedes an `if` statement that uses the walrus operator. Since the assignment is now part of the `if` statement, the separate declaration is no longer needed.

By applying these rules, you can refactor your Python code to be more concise and readable, taking advantage of the walrus operator's ability to combine an assignment with an expression.

### YAML

```yaml
id: use-walrus-operator
language: python
rule:
  follows:
    pattern:
      context: $VAR = $$$EXPR
      selector: expression_statement
  pattern: "if $VAR: $$$B"
fix: |-
  if $VAR := $$$EXPR:
    $$$B
---
id: remove-declaration
language: python
rule:
  pattern:
    context: $VAR = $$$EXPR
    selector: expression_statement
  precedes:
    pattern: "if $VAR: $$$B"
fix: ''
```

### Example

```python
a = foo()

if a:
    do_bar()
```

### Diff

```python
a = foo() # [!code --]

if a: # [!code --]
if a := foo(): # [!code ++]
    do_bar()
```

### Contributed by

Inspired by reddit user [/u/jackerhack](https://www.reddit.com/r/rust/comments/13eg738/comment/kagdklw/?)

## Remove `async` function&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiYXdhaXQgJCQkQ0FMTCIsInJld3JpdGUiOiIkJCRDQUxMICIsImNvbmZpZyI6ImlkOiByZW1vdmUtYXN5bmMtZGVmXG5sYW5ndWFnZTogcHl0aG9uXG5ydWxlOlxuICBwYXR0ZXJuOlxuICAgIGNvbnRleHQ6ICdhc3luYyBkZWYgJEZVTkMoJCQkQVJHUyk6ICQkJEJPRFknXG4gICAgc2VsZWN0b3I6IGZ1bmN0aW9uX2RlZmluaXRpb25cbnRyYW5zZm9ybTpcbiAgUkVNT1ZFRF9CT0RZOlxuICAgIHJld3JpdGU6XG4gICAgICByZXdyaXRlcnM6IFtyZW1vdmUtYXdhaXQtY2FsbF1cbiAgICAgIHNvdXJjZTogJCQkQk9EWVxuZml4OiB8LVxuICBkZWYgJEZVTkMoJCQkQVJHUyk6XG4gICAgJFJFTU9WRURfQk9EWVxucmV3cml0ZXJzOlxuLSBpZDogcmVtb3ZlLWF3YWl0LWNhbGxcbiAgcnVsZTpcbiAgICBwYXR0ZXJuOiAnYXdhaXQgJCQkQ0FMTCdcbiAgZml4OiAkJCRDQUxMXG4iLCJzb3VyY2UiOiJhc3luYyBkZWYgbWFpbjMoKTpcbiAgYXdhaXQgc29tZWNhbGwoMSwgNSkifQ==)

### Description

The `async` keyword in Python is used to define asynchronous functions that can be `await`ed.

In this example, we want to remove the `async` keyword from a function definition and replace it with a synchronous version of the function. We also need to remove the `await` keyword from the function body.

By default, ast-grep will not apply overlapping replacements. This means `await` keywords will not be modified because they are inside the async function body.

However, we can use the [`rewriter`](https://ast-grep.github.io/reference/yaml/rewriter.html) to apply changes inside the matched function body.

### YAML

```yaml
id: remove-async-def
language: python
rule:
  # match async function definition
  pattern:
    context: 'async def $FUNC($$$ARGS): $$$BODY'
    selector: function_definition
rewriters:
# define a rewriter to remove the await keyword
  remove-await-call:
    pattern: 'await $$$CALL'
    fix: $$$CALL # remove await keyword
# apply the rewriter to the function body
transform:
  REMOVED_BODY:
    rewrite:
      rewriters: [remove-await-call]
      source: $$$BODY
fix: |-
  def $FUNC($$$ARGS):
    $REMOVED_BODY
```

### Example

```python
async def main3():
  await somecall(1, 5)
```

### Diff

```python
async def main3(): # [!code --]
  await somecall(1, 5) # [!code --]
def main3(): # [!code ++]
  somecall(1, 5) # [!code ++]
```

### Contributed by

Inspired by the ast-grep issue [#1185](https://github.com/ast-grep/ast-grep/issues/1185)

## Refactor pytest fixtures

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmIGZvbygkWCk6XG4gICRTIiwicmV3cml0ZSI6ImxvZ2dlci5sb2coJE1BVENIKSIsImNvbmZpZyI6ImlkOiBweXRlc3QtdHlwZS1oaW50LWZpeHR1cmVcbmxhbmd1YWdlOiBQeXRob25cbnV0aWxzOlxuICBpcy1maXh0dXJlLWZ1bmN0aW9uOlxuICAgIGtpbmQ6IGZ1bmN0aW9uX2RlZmluaXRpb25cbiAgICBmb2xsb3dzOlxuICAgICAga2luZDogZGVjb3JhdG9yXG4gICAgICBoYXM6XG4gICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgcmVnZXg6IF5maXh0dXJlJFxuICAgICAgICBzdG9wQnk6IGVuZFxuICBpcy10ZXN0LWZ1bmN0aW9uOlxuICAgIGtpbmQ6IGZ1bmN0aW9uX2RlZmluaXRpb25cbiAgICBoYXM6XG4gICAgICBmaWVsZDogbmFtZVxuICAgICAgcmVnZXg6IF50ZXN0X1xuICBpcy1weXRlc3QtY29udGV4dDpcbiAgICAjIFB5dGVzdCBjb250ZXh0IGlzIGEgbm9kZSBpbnNpZGUgYSBweXRlc3RcbiAgICAjIHRlc3QvZml4dHVyZVxuICAgIGluc2lkZTpcbiAgICAgIHN0b3BCeTogZW5kXG4gICAgICBhbnk6XG4gICAgICAgIC0gbWF0Y2hlczogaXMtZml4dHVyZS1mdW5jdGlvblxuICAgICAgICAtIG1hdGNoZXM6IGlzLXRlc3QtZnVuY3Rpb25cbiAgaXMtZml4dHVyZS1hcmc6XG4gICAgIyBGaXh0dXJlIGFyZ3VtZW50cyBhcmUgaWRlbnRpZmllcnMgaW5zaWRlIHRoZSBcbiAgICAjIHBhcmFtZXRlcnMgb2YgYSB0ZXN0L2ZpeHR1cmUgZnVuY3Rpb25cbiAgICBhbGw6XG4gICAgICAtIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgIC0gbWF0Y2hlczogaXMtcHl0ZXN0LWNvbnRleHRcbiAgICAgIC0gaW5zaWRlOlxuICAgICAgICAgIGtpbmQ6IHBhcmFtZXRlcnNcbnJ1bGU6XG4gIG1hdGNoZXM6IGlzLWZpeHR1cmUtYXJnXG4gIHJlZ2V4OiBeZm9vJFxuZml4OiAnZm9vOiBpbnQnXG4iLCJzb3VyY2UiOiJmcm9tIGNvbGxlY3Rpb25zLmFiYyBpbXBvcnQgSXRlcmFibGVcbmZyb20gdHlwaW5nIGltcG9ydCBBbnlcblxuaW1wb3J0IHB5dGVzdFxuZnJvbSBweXRlc3QgaW1wb3J0IGZpeHR1cmVcblxuQHB5dGVzdC5maXh0dXJlKHNjb3BlPVwic2Vzc2lvblwiKVxuZGVmIGZvbygpIC0+IEl0ZXJhYmxlW2ludF06XG4gICAgeWllbGQgNVxuXG5AZml4dHVyZVxuZGVmIGJhcihmb28pIC0+IHN0cjpcbiAgICByZXR1cm4gc3RyKGZvbylcblxuZGVmIHJlZ3VsYXJfZnVuY3Rpb24oZm9vKSAtPiBOb25lOlxuICAgICMgVGhpcyBmdW5jdGlvbiBkb2Vzbid0IHVzZSB0aGUgJ2ZvbycgZml4dHVyZVxuICAgIHByaW50KGZvbylcblxuZGVmIHRlc3RfMShmb28sIGJhcik6XG4gICAgcHJpbnQoZm9vLCBiYXIpXG5cbmRlZiB0ZXN0XzIoYmFyKTpcbiAgICAuLi4ifQ==)

### Description

One of the most commonly used testing framework in Python is [pytest](https://docs.pytest.org/en/8.2.x/). Among other things, it allows the use of [fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).

Fixtures are defined as functions that can be required in test code, or in other fixtures, as an argument. This means that all functions arguments with a given name in a pytest context (test function or fixture) are essentially the same entity. However, not every editor's LSP is able to keep track of this, making refactoring challenging.

Using ast-grep, we can define some rules to match fixture definition and usage without catching similarly named entities in a non-test context.

First, we define utils to select pytest test/fixture functions.

```yaml
utils:
  is-fixture-function:
    kind: function_definition
    follows:
      kind: decorator
      has:
        kind: identifier
        regex: ^fixture$
        stopBy: end
  is-test-function:
    kind: function_definition
    has:
      field: name
      regex: ^test_
```

Pytest fixtures are declared with a decorator `@pytest.fixture`. We match the `function_definition` node that directly follows a `decorator` node. That decorator node must have a `fixture` identifier somewhere. This accounts for different location of the `fixture` node depending on the type of imports and whether the decorator is used as is or called with parameters.

Pytest functions are fairly straightforward to detect, as they always start with `test_` by convention.

The next utils builds onto those two to incrementally:

* Find if a node is inside a pytest context (test/fixture)
* Find if a node is an argument in such a context

```yaml
utils:
  is-pytest-context:
    # Pytest context is a node inside a pytest
    # test/fixture
    inside:
      stopBy: end
      any:
        - matches: is-fixture-function
        - matches: is-test-function
  is-fixture-arg:
    # Fixture arguments are identifiers inside the 
    # parameters of a test/fixture function
    all:
      - kind: identifier
      - inside:
          kind: parameters
      - matches: is-pytest-context
```

Once those utils are declared, you can perform various refactoring on a specific fixture.

The following rule adds a type-hint to a fixture.

```yaml
rule:
  matches: is-fixture-arg
  regex: ^foo$
fix: 'foo: int'
```

This one renames a fixture and all its references.

```yaml
rule:
  kind: identifier
  matches: is-fixture-context
  regex: ^foo$
fix: 'five'
```

### Example

#### Renaming Fixtures

```python {2,6,7,12,13}
@pytest.fixture
def foo() -> int:
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo: int) -> str:
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo: int) -> None:
    assert foo == 5
```

#### Diff

```python {2,6,7,12}
@pytest.fixture
def foo() -> int: # [!code --]
def five() -> int: # [!code ++]
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo: int) -> str: # [!code --]
def some_fixture(five: int) -> str: # [!code ++]
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo: int) -> None: # [!code --]
def test_code(five: int) -> None: # [!code ++]
    assert foo == 5 # [!code --]
    assert five == 5 # [!code ++]
```

#### Type Hinting Fixtures

```python {6,12}
@pytest.fixture
def foo() -> int:
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo) -> str:
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo) -> None:
    assert foo == 5
```

#### Diff

```python {2,6,7,12}
@pytest.fixture
def foo() -> int:
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo) -> str: # [!code --]
def some_fixture(foo: int) -> str: # [!code ++]
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo) -> None: # [!code --]
def test_code(foo: int) -> None: # [!code ++]
    assert foo == 5
```

## Rewrite `Optional[Type]` to `Type | None`&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzaWduYXR1cmUiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IFxuICAgIGNvbnRleHQ6ICdhOiBPcHRpb25hbFskVF0nXG4gICAgc2VsZWN0b3I6IGdlbmVyaWNfdHlwZVxuZml4OiAkVCB8IE5vbmUiLCJzb3VyY2UiOiJkZWYgYShhcmc6IE9wdGlvbmFsW0ludF0pOiBwYXNzIn0=)

### Description

[PEP 604](https://peps.python.org/pep-0604/) recommends that `Type | None` is preferred over `Optional[Type]` for Python 3.10+.

This rule performs such rewriting. Note `Optional[$T]` alone is interpreted as subscripting expression instead of generic type, we need to use [pattern object](/guide/rule-config/atomic-rule.html#pattern-object) to disambiguate it with more context code.

### YAML

```yaml
id: optional-to-none-union
language: python
rule:
  pattern:
    context: 'a: Optional[$T]'
    selector: generic_type
fix: $T | None
```

### Example

```py {1}
def a(arg: Optional[int]): pass
```

### Diff

```py
def a(arg: Optional[int]): pass # [!code --]
def a(arg: int | None): pass # [!code ++]
```

### Contributed by

[Bede Carroll](https://github.com/ast-grep/ast-grep/discussions/1492)

## Recursive Rewrite Type&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicmV3cml0ZXJzOlxyXG4tIGlkOiBvcHRpb25hbFxyXG4gIGxhbmd1YWdlOiBQeXRob25cclxuICBydWxlOlxyXG4gICAgYW55OlxyXG4gICAgLSBwYXR0ZXJuOlxyXG4gICAgICAgIGNvbnRleHQ6ICdhcmc6IE9wdGlvbmFsWyRUWVBFXSdcclxuICAgICAgICBzZWxlY3RvcjogZ2VuZXJpY190eXBlXHJcbiAgICAtIHBhdHRlcm46IE9wdGlvbmFsWyRUWVBFXVxyXG4gIHRyYW5zZm9ybTpcclxuICAgIE5UOlxyXG4gICAgICByZXdyaXRlOiBcclxuICAgICAgICByZXdyaXRlcnM6IFtvcHRpb25hbCwgdW5pb25zXVxyXG4gICAgICAgIHNvdXJjZTogJFRZUEVcclxuICBmaXg6ICROVCB8IE5vbmVcclxuLSBpZDogdW5pb25zXHJcbiAgbGFuZ3VhZ2U6IFB5dGhvblxyXG4gIHJ1bGU6XHJcbiAgICBwYXR0ZXJuOlxyXG4gICAgICBjb250ZXh0OiAnYTogVW5pb25bJCQkVFlQRVNdJ1xyXG4gICAgICBzZWxlY3RvcjogZ2VuZXJpY190eXBlXHJcbiAgdHJhbnNmb3JtOlxyXG4gICAgVU5JT05TOlxyXG4gICAgICByZXdyaXRlOlxyXG4gICAgICAgIHJld3JpdGVyczpcclxuICAgICAgICAgIC0gcmV3cml0ZS11bmlvbnNcclxuICAgICAgICBzb3VyY2U6ICQkJFRZUEVTXHJcbiAgICAgICAgam9pbkJ5OiBcIiB8IFwiXHJcbiAgZml4OiAkVU5JT05TXHJcbi0gaWQ6IHJld3JpdGUtdW5pb25zXHJcbiAgcnVsZTpcclxuICAgIHBhdHRlcm46ICRUWVBFXHJcbiAgICBraW5kOiB0eXBlXHJcbiAgdHJhbnNmb3JtOlxyXG4gICAgTlQ6XHJcbiAgICAgIHJld3JpdGU6IFxyXG4gICAgICAgIHJld3JpdGVyczogW29wdGlvbmFsLCB1bmlvbnNdXHJcbiAgICAgICAgc291cmNlOiAkVFlQRVxyXG4gIGZpeDogJE5UXHJcbnJ1bGU6XHJcbiAga2luZDogdHlwZVxyXG4gIHBhdHRlcm46ICRUUEVcclxudHJhbnNmb3JtOlxyXG4gIE5FV19UWVBFOlxyXG4gICAgcmV3cml0ZTogXHJcbiAgICAgIHJld3JpdGVyczogW29wdGlvbmFsLCB1bmlvbnNdXHJcbiAgICAgIHNvdXJjZTogJFRQRVxyXG5maXg6ICRORVdfVFlQRSIsInNvdXJjZSI6InJlc3VsdHM6ICBPcHRpb25hbFtVbmlvbltMaXN0W1VuaW9uW3N0ciwgZGljdF1dLCBzdHJdXVxuIn0=)

### Description

Suppose we want to transform Python's `Union[T1, T2]` to `T1 | T2` and `Optional[T]` to `T | None`.

By default, ast-grep will only fix the outermost node that matches a pattern and will not rewrite the inner AST nodes inside a match. This avoids unexpected rewriting or infinite rewriting loop.

So if you are using non-recursive rewriter like [this](https://github.com/ast-grep/ast-grep/discussions/1566#discussion-7401382), `Optional[Union[int, str]]` will only be converted to `Union[int, str] | None`. Note the inner `Union[int, str]` is not enabled. This is because the rewriter `optional` matches `Optional[$TYPE]` and rewrite it to `$TYPE | None`. The inner `$TYPE` is not processed.

However, we can apply `rewriters` to inner types recursively. Take the `optional` rewriter as an example, we need to apply rewriters, `optional` and `unions`, **recursively** to `$TYPE` and get a new variable `$NT`.

### YAML

```yml
id: recursive-rewrite-types
language: python
rewriters:
# rewrite Optional[T] to T | None
- id: optional
  rule:
    any:
    - pattern:
        context: 'arg: Optional[$TYPE]'
        selector: generic_type
    - pattern: Optional[$TYPE]
  # recursively apply rewriters to $TYPE
  transform:
    NT:
      rewrite:
        rewriters: [optional, unions]
        source: $TYPE
  # use the new variable $NT
  fix: $NT | None

# similar to Optional, rewrite Union[T1, T2] to T1 | T2
- id: unions
  language: Python
  rule:
    pattern:
      context: 'a: Union[$$$TYPES]'
      selector: generic_type
  transform:
    UNIONS:
      # rewrite all types inside $$$TYPES
      rewrite:
        rewriters: [ rewrite-unions ]
        source: $$$TYPES
        joinBy: " | "
  fix: $UNIONS
- id: rewrite-unions
  rule:
    pattern: $TYPE
    kind: type
  # recursive part
  transform:
    NT:
      rewrite:
        rewriters: [optional, unions]
        source: $TYPE
  fix: $NT

# find all types
rule:
  kind: type
  pattern: $TPE
# apply the recursive rewriters
transform:
  NEW_TYPE:
    rewrite:
      rewriters: [optional, unions]
      source: $TPE
# output
fix: $NEW_TYPE
```

### Example

```python
results:  Optional[Union[List[Union[str, dict]], str]]
```

### Diff

```python
results:  Optional[Union[List[Union[str, dict]], str]] # [!code --]
results:  List[str | dict] | str | None #[!code ++]
```

### Contributed by

Inspired by [steinuil](https://github.com/ast-grep/ast-grep/discussions/1566)

## Rewrite SQLAlchemy with Type Annotations&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiYShudWxsYWJsZT1UcnVlKSIsInJld3JpdGUiOiIxMjMiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6ImtleXdvcmRfYXJndW1lbnQiLCJjb25maWciOiJpZDogcmVtb3ZlLW51bGxhYmxlLWFyZ1xubGFuZ3VhZ2U6IHB5dGhvblxucnVsZTpcbiAgcGF0dGVybjogJFggPSBtYXBwZWRfY29sdW1uKCQkJEFSR1MpXG4gIGFueTpcbiAgICAtIHBhdHRlcm46ICRYID0gbWFwcGVkX2NvbHVtbigkJCRCRUZPUkUsIFN0cmluZywgJCQkTUlELCBudWxsYWJsZT1UcnVlLCAkJCRBRlRFUilcbiAgICAtIHBhdHRlcm46ICRYID0gbWFwcGVkX2NvbHVtbigkJCRCRUZPUkUsIFN0cmluZywgJCQkTUlELCBudWxsYWJsZT1UcnVlKVxucmV3cml0ZXJzOlxuLSBpZDogZmlsdGVyLXN0cmluZy1udWxsYWJsZVxuICBydWxlOlxuICAgIHBhdHRlcm46ICRBUkdcbiAgICBpbnNpZGU6XG4gICAgICBraW5kOiBhcmd1bWVudF9saXN0XG4gICAgYWxsOlxuICAgIC0gbm90OiBcbiAgICAgICAgcGF0dGVybjogU3RyaW5nXG4gICAgLSBub3Q6XG4gICAgICAgIHBhdHRlcm46XG4gICAgICAgICAgY29udGV4dDogYShudWxsYWJsZT1UcnVlKVxuICAgICAgICAgIHNlbGVjdG9yOiBrZXl3b3JkX2FyZ3VtZW50XG4gIGZpeDogJEFSR1xuXG50cmFuc2Zvcm06XG4gIE5FV0FSR1M6XG4gICAgcmV3cml0ZTpcbiAgICAgIHJld3JpdGVyczogW2ZpbHRlci1zdHJpbmctbnVsbGFibGVdXG4gICAgICBzb3VyY2U6ICQkJEFSR1NcbiAgICAgIGpvaW5CeTogJywgJ1xuZml4OiB8LVxuICAkWDogTWFwcGVkW3N0ciB8IE5vbmVdID0gbWFwcGVkX2NvbHVtbigkTkVXQVJHUykiLCJzb3VyY2UiOiJtZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIGRlZmF1bHQ9XCJoZWxsb1wiLCBudWxsYWJsZT1UcnVlKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIG51bGxhYmxlPVRydWUpXG5cbl9tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihcIm1lc3NhZ2VcIiwgU3RyaW5nLCBudWxsYWJsZT1UcnVlKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIG51bGxhYmxlPVRydWUsIHVuaXF1ZT1UcnVlKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihcbiAgU3RyaW5nLCBpbmRleD1UcnVlLCBudWxsYWJsZT1UcnVlLCB1bmlxdWU9VHJ1ZSlcblxuIyBTaG91bGQgbm90IGJlIHRyYW5zZm9ybWVkXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIGRlZmF1bHQ9XCJoZWxsb1wiKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIGRlZmF1bHQ9XCJoZWxsb1wiLCBudWxsYWJsZT1GYWxzZSlcblxubWVzc2FnZSA9IG1hcHBlZF9jb2x1bW4oSW50ZWdlciwgZGVmYXVsdD1cImhlbGxvXCIpIn0=)

### Description

[SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html) recommends using type annotations with `Mapped` type for modern declarative mapping. The `mapped_column()` construct can derive its configuration from [PEP 484](https://peps.python.org/pep-0484/) type annotations.

This rule helps migrate legacy SQLAlchemy code that explicitly uses `String` type and `nullable=True` to the modern type annotation approach using `Mapped[str | None]`.

The key technique demonstrated here is using **rewriters** to selectively filter arguments. The rewriter:

1. Matches each argument inside the `argument_list`
2. Excludes the `String` type argument
3. Excludes the `nullable=True` keyword argument
4. Keeps all other arguments

### YAML

```yaml
id: remove-nullable-arg
language: python
rule:
  pattern: $X = mapped_column($$$ARGS)
  any:
    - pattern: $X = mapped_column($$$BEFORE, String, $$$MID, nullable=True, $$$AFTER)
    - pattern: $X = mapped_column($$$BEFORE, String, $$$MID, nullable=True)
rewriters:
- id: filter-string-nullable
  rule:
    pattern: $ARG
    inside:
      kind: argument_list
    all:
    - not:
        pattern: String
    - not:
        pattern:
          context: a(nullable=True)
          selector: keyword_argument
  fix: $ARG

transform:
  NEWARGS:
    rewrite:
      rewriters: [filter-string-nullable]
      source: $$$ARGS
      joinBy: ', '
fix: |-
  $X: Mapped[str | None] = mapped_column($NEWARGS)
```

### Example

```python {1,3,5,7-8}
message = mapped_column(String, default="hello", nullable=True)

message = mapped_column(String, nullable=True)

_message = mapped_column("message", String, nullable=True)

message = mapped_column(String, nullable=True, unique=True)

message = mapped_column(
  String, index=True, nullable=True, unique=True)

# Should not be transformed
message = mapped_column(String, default="hello")

message = mapped_column(String, default="hello", nullable=False)

message = mapped_column(Integer, default="hello")
```

### Diff

```python
message = mapped_column(String, default="hello", nullable=True) # [!code --]
message: Mapped[str | None] = mapped_column(default="hello") # [!code ++]

message = mapped_column(String, nullable=True) # [!code --]
message: Mapped[str | None] = mapped_column() # [!code ++]

_message = mapped_column("message", String, nullable=True) # [!code --]
_message: Mapped[str | None] = mapped_column("message") # [!code ++]

message = mapped_column(String, nullable=True, unique=True) # [!code --]
message: Mapped[str | None] = mapped_column(unique=True) # [!code ++]

message = mapped_column( # [!code --]
  String, index=True, nullable=True, unique=True) # [!code --]
message: Mapped[str | None] = mapped_column( # [!code ++]
  index=True, unique=True) # [!code ++]
```

### Contributed by

Inspired by [discussion #2319](https://github.com/ast-grep/ast-grep/discussions/2319)

---

---
url: /catalog/ruby.md
---
# Ruby

This page curates a list of example ast-grep rules to check and to rewrite Ruby applications.

## Migrate action\_filter in Ruby on Rails&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1YnkiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiIyBhc3QtZ3JlcCBZQU1MIFJ1bGUgaXMgcG93ZXJmdWwgZm9yIGxpbnRpbmchXG4jIGh0dHBzOi8vYXN0LWdyZXAuZ2l0aHViLmlvL2d1aWRlL3J1bGUtY29uZmlnLmh0bWwjcnVsZVxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogYmVmb3JlX2ZpbHRlciAkJCRBQ1RJT05cbiAgICAtIHBhdHRlcm46IGFyb3VuZF9maWx0ZXIgJCQkQUNUSU9OXG4gICAgLSBwYXR0ZXJuOiBhZnRlcl9maWx0ZXIgJCQkQUNUSU9OXG4gIGhhczpcbiAgICBwYXR0ZXJuOiAkRklMVEVSXG4gICAgZmllbGQ6IG1ldGhvZFxuZml4OiBcbiAgJE5FV19BQ1RJT04gJCQkQUNUSU9OXG50cmFuc2Zvcm06XG4gIE5FV19BQ1RJT046XG4gICAgcmVwbGFjZTpcbiAgICAgIHNvdXJjZTogJEZJTFRFUlxuICAgICAgcmVwbGFjZTogX2ZpbHRlclxuICAgICAgYnk6IF9hY3Rpb24iLCJzb3VyY2UiOiJjbGFzcyBUb2Rvc0NvbnRyb2xsZXIgPCBBcHBsaWNhdGlvbkNvbnRyb2xsZXJcbiAgYmVmb3JlX2ZpbHRlciA6YXV0aGVudGljYXRlXG4gIGFyb3VuZF9maWx0ZXIgOndyYXBfaW5fdHJhbnNhY3Rpb24sIG9ubHk6IDpzaG93XG4gIGFmdGVyX2ZpbHRlciBkbyB8Y29udHJvbGxlcnxcbiAgICBmbGFzaFs6ZXJyb3JdID0gXCJZb3UgbXVzdCBiZSBsb2dnZWQgaW5cIlxuICBlbmRcblxuICBkZWYgaW5kZXhcbiAgICBAdG9kb3MgPSBUb2RvLmFsbFxuICBlbmRcbmVuZFxuIn0=)

### Description

This rule is used to migrate `{before,after,around}_filter` to `{before,after,around}_action` in Ruby on Rails controllers.

These are methods that run before, after or around an action is executed, and they can be used to check permissions, set variables, redirect requests, log events, etc. However, these methods are [deprecated](https://stackoverflow.com/questions/16519828/rails-4-before-filter-vs-before-action) in Rails 5.0 and will be removed in Rails 5.1. `{before,after,around}_action` are the new syntax for the same functionality.

This rule will replace all occurrences of `{before,after,around}_filter` with `{before,after,around}_action` in the controller code.

### YAML

```yaml
id: migration-action-filter
language: ruby
rule:
  any:
    - pattern: before_filter $$$ACTION
    - pattern: around_filter $$$ACTION
    - pattern: after_filter $$$ACTION
  has:
    pattern: $FILTER
    field: method
fix:
  $NEW_ACTION $$$ACTION
transform:
  NEW_ACTION:
    replace:
      source: $FILTER
      replace: _filter
      by: _action
```

### Example

```rb {2-4}
class TodosController < ApplicationController
  before_filter :authenticate
  around_filter :wrap_in_transaction, only: :show
  after_filter do |controller|
    flash[:error] = "You must be logged in"
  end

  def index
    @todos = Todo.all
  end
end
```

### Diff

```rb
class TodosController < ApplicationController
  before_action :authenticate  # [!code --]
  before_filter :authenticate # [!code ++]
  around_action :wrap_in_transaction, only: :show # [!code --]
  around_filter :wrap_in_transaction, only: :show # [!code ++]
  after_action do |controller|  # [!code --]
     flash[:error] = "You must be logged in" # [!code --]
  end # [!code --]
  after_filter do |controller| # [!code ++]
    flash[:error] = "You must be logged in" # [!code ++]
  end # [!code ++]

  def index
    @todos = Todo.all
  end
end
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [Future of Ruby - AST Tooling](https://dev.to/baweaver/future-of-ruby-ast-tooling-9i1).

## Prefer Symbol over Proc&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1YnkiLCJxdWVyeSI6IiRMSVNULnNlbGVjdCB7IHwkVnwgJFYuJE1FVEhPRCB9IiwicmV3cml0ZSI6IiRMSVNULnNlbGVjdCgmOiRNRVRIT0QpIiwiY29uZmlnIjoiaWQ6IHByZWZlci1zeW1ib2wtb3Zlci1wcm9jXG5ydWxlOlxuICBwYXR0ZXJuOiAkTElTVC4kSVRFUiB7IHwkVnwgJFYuJE1FVEhPRCB9XG5sYW5ndWFnZTogUnVieVxuY29uc3RyYWludHM6XG4gIElURVI6XG4gICAgcmVnZXg6ICdtYXB8c2VsZWN0fGVhY2gnXG5maXg6ICckTElTVC4kSVRFUigmOiRNRVRIT0QpJ1xuIiwic291cmNlIjoiWzEsIDIsIDNdLnNlbGVjdCB7IHx2fCB2LmV2ZW4/IH1cbigxLi4xMDApLmVhY2ggeyB8aXwgaS50b19zIH1cbm5vdF9saXN0Lm5vX21hdGNoIHsgfHZ8IHYuZXZlbj8gfVxuIn0=)

### Description

Ruby has a more concise symbol shorthand `&:` to invoke methods.
This rule simplifies `proc` to `symbol`.
This example is inspired by this [dev.to article](https://dev.to/baweaver/future-of-ruby-ast-tooling-9i1).

### YAML

```yaml
id: prefer-symbol-over-proc
language: ruby
rule:
  pattern: $LIST.$ITER { |$V| $V.$METHOD }
constraints:
  ITER:
    regex: 'map|select|each'
fix: '$LIST.$ITER(&:$METHOD)'
```

### Example

```rb {1,2}
[1, 2, 3].select { |v| v.even? }
(1..100).each { |i| i.to_s }
not_list.no_match { |v| v.even? }
```

### Diff

```rb
[1, 2, 3].select { |v| v.even? } # [!code --]
[1, 2, 3].select(&:even?) # [!code ++]
(1..100).each { |i| i.to_s } # [!code --]
(1..100).each(&:to_s) # [!code ++]

not_list.no_match { |v| v.even? }
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim)

## Detect Path Traversal Vulnerability in Rails

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1YnkiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoiYXN0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogcGF0aC10cmF2ZXJzYWxcbm1lc3NhZ2U6IFBvdGVudGlhbCBQYXRoIFRyYXZlcnNhbCB2dWxuZXJhYmlsaXR5IGRldGVjdGVkLiBVc2VyIGlucHV0IGlzIGJlaW5nIHVzZWQgdG8gY29uc3RydWN0IGZpbGUgcGF0aHMgd2l0aG91dCB2YWxpZGF0aW9uLlxuc2V2ZXJpdHk6IGhpbnRcbmxhbmd1YWdlOiBSdWJ5XG5ub3RlOiB8XG4gIFBhdGggVHJhdmVyc2FsIChEaXJlY3RvcnkgVHJhdmVyc2FsKSBvY2N1cnMgd2hlbiB1c2VyIGlucHV0IGlzIHVzZWQgdG8gY29uc3RydWN0IGZpbGUgcGF0aHNcbiAgd2l0aG91dCBwcm9wZXIgdmFsaWRhdGlvbi4gVGhpcyBhbGxvd3MgYXR0YWNrZXJzIHRvIGFjY2VzcyBmaWxlcyBvdXRzaWRlIHRoZSBpbnRlbmRlZCBkaXJlY3RvcnkuXG4gIFZhbGlkYXRlIGFuZCBzYW5pdGl6ZSBmaWxlIHBhdGhzLCBhbmQgdXNlIEZpbGUuYmFzZW5hbWUoKSBvciBzaW1pbGFyIGZ1bmN0aW9ucy5cblxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogUmFpbHMucm9vdC5qb2luKCQkJCwgJFZBUiwgJCQkKVxuICAgIC0gcGF0dGVybjogRmlsZS5qb2luKCQkJCwgJFZBUiwgJCQkKVxuICAgIC0gcGF0dGVybjogc2VuZF9maWxlICRWQVIiLCJzb3VyY2UiOiIjIOaknOWHuuOBleOCjOOCi+OCs+ODvOODieS+i1xuIyDjg5Hjgr/jg7zjg7MxOiBSYWlscy5yb290LmpvaW4gd2l0aCB2YXJpYWJsZVxuUmFpbHMucm9vdC5qb2luKCd1cGxvYWRzJywgcGFyYW1zWzpmaWxlbmFtZV0pXG5SYWlscy5yb290LmpvaW4oJ2RhdGEnLCB1c2VyX2lucHV0LCAnZmlsZS50eHQnKVxuXG4jIOODkeOCv+ODvOODszI6IEZpbGUuam9pbiB3aXRoIHZhcmlhYmxlXG5GaWxlLmpvaW4oJy92YXIvd3d3JywgcGFyYW1zWzpwYXRoXSlcbkZpbGUuam9pbihiYXNlX3BhdGgsIHVzZXJfaWQsIGZpbGVuYW1lKVxuXG4jIOODkeOCv+ODvOODszM6IHNlbmRfZmlsZSB3aXRoIHZhcmlhYmxlXG5zZW5kX2ZpbGUgcGFyYW1zWzpmaWxlXVxuc2VuZF9maWxlIHVzZXIuZG9jdW1lbnRfcGF0aCJ9)

### Description

Path Traversal (Directory Traversal) occurs when user input is used to construct file paths without proper validation. This allows attackers to access files outside the intended directory by using special characters like `../` to navigate the filesystem.

This rule detects common path traversal patterns in Rails applications where user-controlled variables are used in:

* `Rails.root.join()` - Building file paths relative to the Rails application root
* `File.join()` - Constructing file paths
* `send_file` - Sending files to users

To prevent path traversal vulnerabilities, always validate and sanitize file paths, use `File.basename()` to extract only the filename, or use allowlists for permitted files.

### YAML

```yaml
id: path-traversal
message: Potential Path Traversal vulnerability detected. User input is being used to construct file paths without validation.
severity: hint
language: Ruby
note: |
  Path Traversal (Directory Traversal) occurs when user input is used to construct file paths
  without proper validation. This allows attackers to access files outside the intended directory.
  Validate and sanitize file paths, and use File.basename() or similar functions.

rule:
  any:
    - pattern: Rails.root.join($$$, $VAR, $$$)
    - pattern: File.join($$$, $VAR, $$$)
    - pattern: send_file $VAR
```

### Example

```rb {2,3,6,7,10,11}
# Pattern 1: Rails.root.join with variable
Rails.root.join('uploads', params[:filename])
Rails.root.join('data', user_input, 'file.txt')

# Pattern 2: File.join with variable
File.join('/var/www', params[:path])
File.join(base_path, user_id, filename)

# Pattern 3: send_file with variable
send_file params[:file]
send_file user.document_path
```

### Contributed by

[sora fs0414](https://x.com/_fs0414) from this [blog post](https://fs0414.hatenablog.com/entry/2025/11/02/032114)

---

---
url: /catalog/rust.md
---
# Rust

This page curates a list of example ast-grep rules to check and to rewrite Rust applications.

## Avoid Duplicated Exports

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIGFsbDpcbiAgICAgLSBwYXR0ZXJuOiBwdWIgdXNlICRCOjokQztcbiAgICAgLSBpbnNpZGU6XG4gICAgICAgIGtpbmQ6IHNvdXJjZV9maWxlXG4gICAgICAgIGhhczpcbiAgICAgICAgICBwYXR0ZXJuOiBwdWIgbW9kICRBO1xuICAgICAtIGhhczpcbiAgICAgICAgcGF0dGVybjogJEFcbiAgICAgICAgc3RvcEJ5OiBlbmQiLCJzb3VyY2UiOiJwdWIgbW9kIGZvbztcbnB1YiB1c2UgZm9vOjpGb287XG5wdWIgdXNlIGZvbzo6QTo6QjtcblxuXG5wdWIgdXNlIGFhYTo6QTtcbnB1YiB1c2Ugd29vOjpXb287In0=)

### Description

Generally, we don't encourage the use of re-exports.

However, sometimes, to keep the interface exposed by a lib crate tidy, we use re-exports to shorten the path to specific items.
When doing so, a pitfall is to export a single item under two different names.

Consider:

```rs
pub mod foo;
pub use foo::Foo;
```

The issue with this code, is that `Foo` is now exposed under two different paths: `Foo`, `foo::Foo`.

This unnecessarily increases the surface of your API.
It can also cause issues on the client side. For example, it makes the usage of auto-complete in the IDE more involved.

Instead, ensure you export only once with `pub`.

### YAML

```yaml
id: avoid-duplicate-export
language: rust
rule:
  all:
     - pattern: pub use $B::$C;
     - inside:
        kind: source_file
        has:
          pattern: pub mod $A;
     - has:
        pattern: $A
        stopBy: end
```

### Example

```rs {2,3}
pub mod foo;
pub use foo::Foo;
pub use foo::A::B;


pub use aaa::A;
pub use woo::Woo;
```

### Contributed by

Julius Lungys([voidpumpkin](https://github.com/voidpumpkin))

## Beware of char offset when iterate over a string&#x20;

* [Playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoicnVzdCIsInF1ZXJ5IjoiJEEuY2hhcnMoKS5lbnVtZXJhdGUoKSIsInJld3JpdGUiOiIkQS5jaGFyX2luZGljZXMoKSIsImNvbmZpZyI6IiIsInNvdXJjZSI6ImZvciAoaSwgY2hhcikgaW4gc291cmNlLmNoYXJzKCkuZW51bWVyYXRlKCkge1xuICAgIHByaW50bG4hKFwiQm9zaGVuIGlzIGFuZ3J5IDopXCIpO1xufSJ9)

### Description

It's a common pitfall in Rust that counting *character offset* is not the same as counting *byte offset* when iterating through a string. Rust string is represented by utf-8 byte array, which is a variable-length encoding scheme.

`chars().enumerate()` will yield the character offset, while [`char_indices()`](https://doc.rust-lang.org/std/primitive.str.html#method.char_indices) will yield the byte offset.

```rs
let yes = "y̆es";
let mut char_indices = yes.char_indices();
assert_eq!(Some((0, 'y')), char_indices.next()); // not (0, 'y̆')
assert_eq!(Some((1, '\u{0306}')), char_indices.next());
// note the 3 here - the last character took up two bytes
assert_eq!(Some((3, 'e')), char_indices.next());
assert_eq!(Some((4, 's')), char_indices.next());
```

Depending on your use case, you may want to use `char_indices()` instead of `chars().enumerate()`.

### Pattern

```shell
ast-grep -p '$A.chars().enumerate()' \
   -r '$A.char_indices()' \
   -l rs
```

### Example

```rs {1}
for (i, char) in source.chars().enumerate() {
    println!("Boshen is angry :)");
}
```

### Diff

```rs
for (i, char) in source.chars().enumerate() { // [!code --]
for (i, char) in source.char_indices() { // [!code ++]
    println!("Boshen is angry :)");
}
```

### Contributed by

Inspired by [Boshen's Tweet](https://x.com/boshen_c/status/1719033308682870891)

![Boshen's footgun](https://pbs.twimg.com/media/F9s7mJHaYAEndnY?format=jpg\&name=medium)

## Get number of digits in a `usize`&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoicnVzdCIsInF1ZXJ5IjoiJE5VTS50b19zdHJpbmcoKS5jaGFycygpLmNvdW50KCkiLCJyZXdyaXRlIjoiJE5VTS5jaGVja2VkX2lsb2cxMCgpLnVud3JhcF9vcigwKSArIDEiLCJjb25maWciOiIjIFlBTUwgUnVsZSBpcyBtb3JlIHBvd2VyZnVsIVxuIyBodHRwczovL2FzdC1ncmVwLmdpdGh1Yi5pby9ndWlkZS9ydWxlLWNvbmZpZy5odG1sI3J1bGVcbnJ1bGU6XG4gIGFueTpcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUubG9nKCRBKVxuICAgIC0gcGF0dGVybjogY29uc29sZS5kZWJ1ZygkQSlcbmZpeDpcbiAgbG9nZ2VyLmxvZygkQSkiLCJzb3VyY2UiOiJsZXQgd2lkdGggPSAobGluZXMgKyBudW0pLnRvX3N0cmluZygpLmNoYXJzKCkuY291bnQoKTsifQ==)

### Description

Getting the number of digits in a usize number can be useful for various purposes, such as counting the column width of line numbers in a text editor or formatting the output of a number with commas or spaces.

A common but inefficient way of getting the number of digits in a `usize` number is to use `num.to_string().chars().count()`. This method converts the number to a string, iterates over its characters, and counts them. However, this method involves allocating a new string, which can be costly in terms of memory and time.

A better alternative is to use [`checked_ilog10`](https://doc.rust-lang.org/std/primitive.usize.html#method.checked_ilog10).

```rs
num.checked_ilog10().unwrap_or(0) + 1
```

The snippet above computes the integer logarithm base 10 of the number and adds one. This snippet does not allocate any memory and is faster than the string conversion approach. The [efficient](https://doc.rust-lang.org/src/core/num/int_log10.rs.html) `checked_ilog10` function returns an `Option<usize>` that is `Some(log)` if the number is positive and `None` if the number is zero. The `unwrap_or(0)` function returns the value inside the option or `0` if the option is `None`.

### Pattern

```shell
ast-grep -p '$NUM.to_string().chars().count()' \
   -r '$NUM.checked_ilog10().unwrap_or(0) + 1' \
   -l rs
```

### Example

```rs {1}
let width = (lines + num).to_string().chars().count();
```

### Diff

```rs
let width = (lines + num).to_string().chars().count(); // [!code --]
let width = (lines + num).checked_ilog10().unwrap_or(0) + 1; // [!code ++]
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [dogfooding ast-grep](https://github.com/ast-grep/ast-grep/issues/550)

## Unsafe Function Without Unsafe Block

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IntcbiAgZGVzY3JpcHRpb24gPSAkQVxufSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6ImJpbmRpbmciLCJjb25maWciOiIgIGlkOiByZWR1bmRhbnQtdW5zYWZlLWZ1bmN0aW9uXG4gIGxhbmd1YWdlOiBydXN0XG4gIHNldmVyaXR5OiBlcnJvclxuICBtZXNzYWdlOiBVbnNhZmUgZnVuY3Rpb24gd2l0aG91dCB1bnNhZmUgYmxvY2sgaW5zaWRlXG4gIG5vdGU6IHxcbiAgICBDb25zaWRlciB3aGV0aGVyIHRoaXMgZnVuY3Rpb24gbmVlZHMgdG8gYmUgbWFya2VkIHVuc2FmZSBcbiAgICBvciBpZiB1bnNhZmUgb3BlcmF0aW9ucyBzaG91bGQgYmUgd3JhcHBlZCBpbiBhbiB1bnNhZmUgYmxvY2tcbiAgcnVsZTpcbiAgICBhbGw6XG4gICAgICAtIGtpbmQ6IGZ1bmN0aW9uX2l0ZW1cbiAgICAgIC0gaGFzOlxuICAgICAgICAgIGtpbmQ6IGZ1bmN0aW9uX21vZGlmaWVyc1xuICAgICAgICAgIHJlZ2V4OiBcIl51bnNhZmVcIlxuICAgICAgLSBub3Q6XG4gICAgICAgICAgaGFzOlxuICAgICAgICAgICAga2luZDogdW5zYWZlX2Jsb2NrXG4gICAgICAgICAgICBzdG9wQnk6IGVuZCIsInNvdXJjZSI6IiAgLy8gU2hvdWxkIG1hdGNoIC0gdW5zYWZlIGZ1bmN0aW9uIHdpdGhvdXQgdW5zYWZlIGJsb2NrIChubyByZXR1cm4gdHlwZSlcbiAgdW5zYWZlIGZuIHJlZHVuZGFudF91bnNhZmUoKSB7XG4gICAgICBwcmludGxuIShcIk5vIHVuc2FmZSBvcGVyYXRpb25zIGhlcmVcIik7XG4gIH1cblxuICAvLyBTaG91bGQgbWF0Y2ggLSB1bnNhZmUgZnVuY3Rpb24gd2l0aCByZXR1cm4gdHlwZSwgbm8gdW5zYWZlIGJsb2NrXG4gIHVuc2FmZSBmbiByZWR1bmRhbnRfd2l0aF9yZXR1cm4oKSAtPiBpMzIge1xuICAgICAgbGV0IHggPSA1O1xuICAgICAgeCArIDEwXG4gIH1cblxuICAvLyBTaG91bGQgbWF0Y2ggLSB1bnNhZmUgZnVuY3Rpb24gd2l0aCBjb21wbGV4IHJldHVybiB0eXBlXG4gIHVuc2FmZSBmbiByZWR1bmRhbnRfY29tcGxleF9yZXR1cm4oKSAtPiBSZXN1bHQ8U3RyaW5nLCBzdGQ6OmlvOjpFcnJvcj4ge1xuICAgICAgT2soU3RyaW5nOjpmcm9tKFwic2FmZSBvcGVyYXRpb25cIikpXG4gIH1cblxuICAvLyBTaG91bGQgTk9UIG1hdGNoIC0gdW5zYWZlIGZ1bmN0aW9uIHdpdGggdW5zYWZlIGJsb2NrXG4gIHVuc2FmZSBmbiBwcm9wZXJfdW5zYWZlKCkgLT4gKmNvbnN0IGkzMiB7XG4gICAgICB1bnNhZmUge1xuICAgICAgICAgIGxldCBwdHIgPSAweDEyMzQgYXMgKmNvbnN0IGkzMjtcbiAgICAgICAgICBwdHJcbiAgICAgIH1cbiAgfVxuXG4gIC8vIFNob3VsZCBtYXRjaCAtIHVuc2FmZSBhc3luYyBmdW5jdGlvbiB3aXRob3V0IHVuc2FmZSBibG9ja1xuICB1bnNhZmUgYXN5bmMgZm4gYXN5bmNfcmVkdW5kYW50KCkgLT4gaTMyIHtcbiAgICAgIDQyXG4gIH1cblxuICAvLyBTaG91bGQgbWF0Y2ggLSB1bnNhZmUgY29uc3QgZnVuY3Rpb25cbiAgdW5zYWZlIGNvbnN0IGZuIGNvbnN0X3JlZHVuZGFudCgpIC0+IGkzMiB7XG4gICAgICAxMDBcbiAgfVxuXG4gIC8vIFNob3VsZCBOT1QgbWF0Y2ggLSByZWd1bGFyIGZ1bmN0aW9uXG4gIGZuIHJlZ3VsYXJfZnVuY3Rpb24oKSAtPiBpMzIge1xuICAgICAgNDJcbiAgfSJ9)

### Description

This rule detects functions marked with the `unsafe` keyword that do not contain any `unsafe` blocks in their body.

When a function is marked `unsafe`, it indicates that the function contains operations that the compiler cannot verify as safe. However, if the function body doesn't contain any `unsafe` blocks, it may be unnecessarily marked as `unsafe`. This could be a sign that:

1. The function should not be marked `unsafe` if it doesn't perform any unsafe operations
2. Or if there are unsafe operations, they should be explicitly wrapped in `unsafe` blocks for clarity

This rule helps identify such cases so developers can review whether the `unsafe` marker is truly necessary or if the code needs to be refactored.

### YAML

```yaml
id: redundant-unsafe-function
language: rust
severity: error
message: Unsafe function without unsafe block inside
note: |
  Consider whether this function needs to be marked unsafe 
  or if unsafe operations should be wrapped in an unsafe block
rule:
  all:
    - kind: function_item
    - has:
        kind: function_modifiers
        regex: "^unsafe"
    - not:
        has:
          kind: unsafe_block
          stopBy: end
```

### Example

```rs {2,7,12,24,29}
// Should match - unsafe function without unsafe block (no return type)
unsafe fn redundant_unsafe() {
    println!("No unsafe operations here");
}

// Should match - unsafe function with return type, no unsafe block
unsafe fn redundant_with_return() -> i32 {
    let x = 5;
    x + 10
}

// Should match - unsafe function with complex return type
unsafe fn redundant_complex_return() -> Result<String, std::io::Error> {
    Ok(String::from("safe operation"))
}

// Should NOT match - unsafe function with unsafe block
unsafe fn proper_unsafe() -> *const i32 {
    unsafe {
        let ptr = 0x1234 as *const i32;
        ptr
    }
}

// Should match - unsafe async function without unsafe block
unsafe async fn async_redundant() -> i32 {
    42
}

// Should match - unsafe const function
unsafe const fn const_redundant() -> i32 {
    100
}

// Should NOT match - regular function
fn regular_function() -> i32 {
    42
}
```

### Contributed by

Inspired by [@hd\_nvim's Tweet](https://x.com/hd_nvim/status/1992810384072585397?s=20)

## Rust 2024 Let-Chain Candidate

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6ImlkOiBydXN0LTIwMjQtbGV0LWNoYWluLWNhbmRpZGF0ZVxubGFuZ3VhZ2U6IFJ1c3RcbnNldmVyaXR5OiBoaW50XG5tZXNzYWdlOiBuZXN0ZWQgaWYgY29uZGl0aW9ucyBjYW4gYmUgY29sbGFwc2VkIGludG8gYSBSdXN0IDIwMjQgbGV0LWNoYWluXG5cbnV0aWxzOlxuICAjIFRoZSBuZXN0ZWQgaWYgc3RhdGVtZW50IHNob3VsZCBiZSB0aGUgb25seSBzdGF0ZW1lbnQgaW4gaXRzIGJsb2NrLlxuICBzb2xlLWNoaWxkOlxuICAgIGFsbDpcbiAgICAgIC0gbnRoQ2hpbGQ6IDFcbiAgICAgIC0gbnRoQ2hpbGQ6XG4gICAgICAgICAgcG9zaXRpb246IDFcbiAgICAgICAgICByZXZlcnNlOiB0cnVlXG5cbiAgIyBMZXQtY2hhaW5zIGNhbiBvbmx5IHByZXNlcnZlIGJlaGF2aW9yIHdoZW4gdGhlcmUgaXMgbm8gZWxzZSBicmFuY2guXG4gIGlmLW5vLWVsc2U6XG4gICAga2luZDogaWZfZXhwcmVzc2lvblxuICAgIG5vdDpcbiAgICAgIGhhczpcbiAgICAgICAgZmllbGQ6IGFsdGVybmF0aXZlXG4gICAgICAgIGtpbmQ6IGVsc2VfY2xhdXNlXG5cbiAgIyBBdCBsZWFzdCBvbmUgc2lkZSBvZiB0aGUgbmVzdGVkIGlmIHBhaXIgbXVzdCBiZSBhbiBpZiBsZXQuXG4gIGlmLWxldC1uby1lbHNlOlxuICAgIG1hdGNoZXM6IGlmLW5vLWVsc2VcbiAgICBoYXM6XG4gICAgICBmaWVsZDogY29uZGl0aW9uXG4gICAgICBraW5kOiBsZXRfY29uZGl0aW9uXG5cbiAgIyBNYXRjaCBhbiBpbm5lciBpZiB0aGF0IGlzIHRoZSBzb2xlIHN0YXRlbWVudCBpbnNpZGUgdGhlIG91dGVyIGlmIGJsb2NrLlxuICBzb2xlLWlubmVyLWlmLXN0bXQ6XG4gICAga2luZDogZXhwcmVzc2lvbl9zdGF0ZW1lbnRcbiAgICBtYXRjaGVzOiBzb2xlLWNoaWxkXG4gICAgaGFzOlxuICAgICAgbWF0Y2hlczogaWYtbm8tZWxzZVxuXG4gICMgU2FtZSBhcyBhYm92ZSwgYnV0IHJlcXVpcmUgdGhlIGlubmVyIGlmIHRvIGJlIGFuIGlmIGxldC5cbiAgc29sZS1pbm5lci1pZi1sZXQtc3RtdDpcbiAgICBraW5kOiBleHByZXNzaW9uX3N0YXRlbWVudFxuICAgIG1hdGNoZXM6IHNvbGUtY2hpbGRcbiAgICBoYXM6XG4gICAgICBtYXRjaGVzOiBpZi1sZXQtbm8tZWxzZVxuXG5ydWxlOlxuICAjIFN0YXJ0IGZyb20gYW4gb3V0ZXIgaWYgd2l0aG91dCBhbiBlbHNlIGJyYW5jaC5cbiAgbWF0Y2hlczogaWYtbm8tZWxzZVxuICAjIEl0cyBkaXJlY3QgY29uc2VxdWVuY2UgYmxvY2sgbXVzdCBjb250YWluIGV4YWN0bHkgb25lIGlubmVyIGlmIHN0YXRlbWVudC5cbiAgaGFzOlxuICAgIGZpZWxkOiBjb25zZXF1ZW5jZVxuICAgIGtpbmQ6IGJsb2NrXG4gICAgaGFzOlxuICAgICAgbWF0Y2hlczogc29sZS1pbm5lci1pZi1zdG10XG4gICMgRWl0aGVyIHRoZSBvdXRlciBjb25kaXRpb24gb3IgdGhlIGlubmVyIGNvbmRpdGlvbiBtdXN0IGJpbmQgd2l0aCBpZiBsZXQuXG4gIGFueTpcbiAgICAtIG1hdGNoZXM6IGlmLWxldC1uby1lbHNlXG4gICAgLSBoYXM6XG4gICAgICAgIGZpZWxkOiBjb25zZXF1ZW5jZVxuICAgICAgICBraW5kOiBibG9ja1xuICAgICAgICBoYXM6XG4gICAgICAgICAgbWF0Y2hlczogc29sZS1pbm5lci1pZi1sZXQtc3RtdCIsInNvdXJjZSI6ImZuIGhhbmRsZV9yZXF1ZXN0KHVzZXI6IE9wdGlvbjxVc2VyPiwgY2ZnOiBDb25maWcpIHtcbiAgICBpZiBsZXQgU29tZSh1c2VyKSA9IHVzZXIge1xuICAgICAgICBpZiB1c2VyLmlzX2FjdGl2ZSgpIHtcbiAgICAgICAgICAgIGdyYW50X2FjY2Vzcyh1c2VyKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIGlmIGNmZyEoZGVidWdfYXNzZXJ0aW9ucykge1xuICAgICAgICBpZiBsZXQgU29tZShwYXRoKSA9IGNmZy5sb2dfcGF0aCgpIHtcbiAgICAgICAgICAgIGVuYWJsZV9maWxlX2xvZ2dpbmcocGF0aCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBPSzogdGhlIGlubmVyIGlmIGlzIG5vdCB0aGUgb25seSBzdGF0ZW1lbnQgaW4gdGhlIGJsb2NrLlxuICAgIGlmIGxldCBTb21lKHVzZXIpID0gY3VycmVudF91c2VyKCkge1xuICAgICAgICBhdWRpdCgmdXNlcik7XG4gICAgICAgIGlmIHVzZXIuaXNfYWRtaW4oKSB7XG4gICAgICAgICAgICBncmFudF9hZG1pbih1c2VyKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIC8vIE9LOiB0aGlzIGhhcyBhbiBlbHNlIGJyYW5jaC5cbiAgICBpZiBsZXQgU29tZShwYXRoKSA9IGNmZy5jYWNoZV9wYXRoKCkge1xuICAgICAgICBpZiBwYXRoLmV4aXN0cygpIHtcbiAgICAgICAgICAgIGxvYWRfY2FjaGUocGF0aCk7XG4gICAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgICByZWJ1aWxkX2NhY2hlKCk7XG4gICAgfVxufSJ9)

### Description

Rust 2024 supports [`let` conditions in boolean chains](https://doc.rust-lang.org/edition-guide/rust-2024/let-chains.html), so an `if let` wrapping a single nested `if` can often be collapsed into one condition.

This rule reports the outer `if` only when neither branch has an `else`, the outer block contains a single inner `if` statement, and either the outer or inner condition is an `if let`.

### YAML

```yaml
id: rust-2024-let-chain-candidate
language: Rust
severity: hint
message: nested if conditions can be collapsed into a Rust 2024 let-chain

utils:
  # The nested if statement should be the only statement in its block.
  sole-child:
    all:
      - nthChild: 1
      - nthChild:
          position: 1
          reverse: true

  # Let-chains can only preserve behavior when there is no else branch.
  if-no-else:
    kind: if_expression
    not:
      has:
        field: alternative
        kind: else_clause

  # At least one side of the nested if pair must be an if let.
  if-let-no-else:
    matches: if-no-else
    has:
      field: condition
      kind: let_condition

  # Match an inner if that is the sole statement inside the outer if block.
  sole-inner-if-stmt:
    kind: expression_statement
    matches: sole-child
    has:
      matches: if-no-else

  # Same as above, but require the inner if to be an if let.
  sole-inner-if-let-stmt:
    kind: expression_statement
    matches: sole-child
    has:
      matches: if-let-no-else

rule:
  # Start from an outer if without an else branch.
  matches: if-no-else
  # Its direct consequence block must contain exactly one inner if statement.
  has:
    field: consequence
    kind: block
    has:
      matches: sole-inner-if-stmt
  # Either the outer condition or the inner condition must bind with if let.
  any:
    - matches: if-let-no-else
    - has:
        field: consequence
        kind: block
        has:
          matches: sole-inner-if-let-stmt
```

### Example

```rs {2,8}
fn handle_request(user: Option<User>, cfg: Config) {
    if let Some(user) = user {
        if user.is_active() {
            grant_access(user);
        }
    }

    if cfg!(debug_assertions) {
        if let Some(path) = cfg.log_path() {
            enable_file_logging(path);
        }
    }

    // OK: the inner if is not the only statement in the block.
    if let Some(user) = current_user() {
        audit(&user);
        if user.is_admin() {
            grant_admin(user);
        }
    }

    // OK: this has an else branch.
    if let Some(path) = cfg.cache_path() {
        if path.exists() {
            load_cache(path);
        }
    } else {
        rebuild_cache();
    }
}
```

### Contributed by

[Codex](https://openai.com/codex/) (GPT-5.5 high fast)

## Rewrite `indoc!` macro&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoicnVzdCIsInF1ZXJ5IjoiaW5kb2MhIHsgciNcIiQkJEFcIiMgfSIsInJld3JpdGUiOiJgJCQkQWAiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTogXG4gYW55OlxuIC0gcGF0dGVybjogJFYgPT09ICRTRU5TRVRJVkVXT1JEXG4gLSBwYXR0ZXJuOiAkU0VOU0VUSVZFV09SRCA9PT0gJFZcbmNvbnN0cmFpbnRzOlxuICBTRU5TRVRJVkVXT1JEOlxuICAgIHJlZ2V4OiBwYXNzd29yZCIsInNvdXJjZSI6ImZuIG1haW4oKSB7XG4gICAgaW5kb2MhIHtyI1wiXG4gICAgICAgIC5mb28ge1xuICAgICAgICAgICAgb3JkZXI6IDE7XG4gICAgICAgIH1cbiAgICBcIiN9O1xufSJ9)

### Description

This example, created from [a Tweet](https://x.com/zack_overflow/status/1885065128590401551), shows a refactoring operation being performed on Rust source code. The changes involve removing `indoc!` macro declarations while preserving the CSS-like content within them.

Previously, the same refactor is implemented by a *unreadable monster regex* in vim syntax.

:::details Click to see the original regex (neovim, btw)

```vimscript
:%s/\v(indoc!|)(| )([|\{)r#"(([^#]+|\n+)+)"#/`\4`
```

I have to confess that I don't understand this regex even if I use neovim, btw.

Let Claude break it down piece by piece:

* `:%s/` - Vim substitution command for all lines
* `\v` - Very magic mode in vim for simpler regex syntax
* `(indoc!|)` - First capture group: matches either "indoc!" or nothing
* `(| )` - Second capture group: matches either empty string or a space
* `([|\{)` - Third capture group: matches either `[` or `{`
* `r#"` - Matches literal `r#"` (Rust raw string delimiter)
* `(([^#]+|\n+)+)` - Fourth capture group (nested):
  * `[^#]+` - One or more non-# characters
  * `|\n+` - OR one or more newlines
  * Outer `()+` makes this repeat one or more times
* `"#` - Matches the closing raw string delimiter
* \`\4\` - Replaces with the fourth capture group wrapped in backticks

This regex is designed to find Rust raw string literals (possibly wrapped in `indoc!` macro), capture their content, and replace the entire match with just the content wrapped in backticks. It's more precise than my previous explanation and matches the pattern you're showing.

:::

### Pattern

```shell
ast-grep --pattern 'indoc! { r#"$$$A"# }' --rewrite '`$$$A`' sgtest.rs
```

### Example

```rs {2-6}
fn main() {
    indoc! {r#"
        .foo {
            order: 1;
        }
    "#};
}
```

### Diff

```rs
fn main() {
    indoc! {r#" // [!code --]
    `.foo {    // [!code ++]
        order: 1;
    }
    "#}; // [!code --]
        `; // [!code ++]
}
```

### Contributed by

[Zack in SF](https://x.com/zack_overflow)

---

---
url: /catalog/typescript.md
---
# TypeScript

This page curates a list of example ast-grep rules to check and to rewrite TypeScript applications.
Check out the [Repository of ESLint rules](https://github.com/ast-grep/eslint/) recreated with ast-grep.

:::danger TypeScript and TSX are different.
TypeScript is a typed JavaScript extension and TSX is a further extension that allows JSX elements.
They need different parsers because of [conflicting syntax](https://www.typescriptlang.org/docs/handbook/jsx.html#the-as-operator).

However, you can use the [`languageGlobs`](/reference/sgconfig.html#languageglobs) option to force ast-grep to use parse `.ts` files as TSX.
:::

## Find Import File without Extension

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoibGFuZ3VhZ2U6IFwianNcIlxucnVsZTpcbiAgcmVnZXg6IFwiL1teLl0rW14vXSRcIiAgXG4gIGtpbmQ6IHN0cmluZ19mcmFnbWVudFxuICBhbnk6XG4gICAgLSBpbnNpZGU6XG4gICAgICAgIHN0b3BCeTogZW5kXG4gICAgICAgIGtpbmQ6IGltcG9ydF9zdGF0ZW1lbnRcbiAgICAtIGluc2lkZTpcbiAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAgICAga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgIGhhczpcbiAgICAgICAgICBmaWVsZDogZnVuY3Rpb25cbiAgICAgICAgICByZWdleDogXCJeaW1wb3J0JFwiXG4iLCJzb3VyY2UiOiJpbXBvcnQgYSwge2IsIGMsIGR9IGZyb20gXCIuL2ZpbGVcIjtcbmltcG9ydCBlIGZyb20gXCIuL290aGVyX2ZpbGUuanNcIjtcbmltcG9ydCBcIi4vZm9sZGVyL1wiO1xuaW1wb3J0IHt4fSBmcm9tIFwicGFja2FnZVwiO1xuaW1wb3J0IHt5fSBmcm9tIFwicGFja2FnZS93aXRoL3BhdGhcIjtcblxuaW1wb3J0KFwiLi9keW5hbWljMVwiKTtcbmltcG9ydChcIi4vZHluYW1pYzIuanNcIik7XG5cbm15X2Z1bmMoXCIuL3VucmVsYXRlZF9wYXRoX3N0cmluZ1wiKVxuXG4ifQ==)

### Description

In ECMAScript modules (ESM), the module specifier must include the file extension, such as `.js` or `.mjs`, when importing local or absolute modules. This is because ESM does not perform any automatic file extension resolution, unlike CommonJS modules tools such as Webpack or Babel. This behavior matches how import behaves in browser environments, and is specified by the [ESM module spec](https://stackoverflow.com/questions/66375075/node-14-ecmascript-modules-import-modules-without-file-extensions).

The rule finds all imports (static and dynamic) for files without a file extension.

### YAML

```yaml
id: find-import-file
language: js
rule:
  regex: "/[^.]+[^/]$"
  kind: string_fragment
  any:
    - inside:
        stopBy: end
        kind: import_statement
    - inside:
        stopBy: end
        kind: call_expression
        has:
          field: function
          regex: "^import$"
```

### Example

```ts {1,5,7}
import a, {b, c, d} from "./file";
import e from "./other_file.js";
import "./folder/";
import {x} from "package";
import {y} from "package/with/path";

import("./dynamic1");
import("./dynamic2.js");

my_func("./unrelated_path_string")
```

### Contributed by

[DasSurma](https://twitter.com/DasSurma) in [this tweet](https://x.com/DasSurma/status/1706213303331029277).

## Migrate XState to v5 from v4&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImlmICgkQSkgeyAkJCRCIH0iLCJyZXdyaXRlIjoiaWYgKCEoJEEpKSB7XG4gICAgcmV0dXJuO1xufVxuJCQkQiIsImNvbmZpZyI6InV0aWxzOlxuICBGUk9NX1hTVEFURTogeyBraW5kOiBpbXBvcnRfc3RhdGVtZW50LCBoYXM6IHsga2luZDogc3RyaW5nLCByZWdleDogeHN0YXRlIH0gfVxuICBYU1RBVEVfRVhQT1JUOlxuICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICBpbnNpZGU6IHsgaGFzOiB7IG1hdGNoZXM6IEZST01fWFNUQVRFIH0sIHN0b3BCeTogZW5kIH1cbnJ1bGU6IHsgcmVnZXg6IF5NYWNoaW5lfGludGVycHJldCQsIHBhdHRlcm46ICRJTVBPUlQsIG1hdGNoZXM6IFhTVEFURV9FWFBPUlQgfVxudHJhbnNmb3JtOlxuICBTVEVQMTogXG4gICAgcmVwbGFjZToge2J5OiBjcmVhdGUkMSwgcmVwbGFjZTogKE1hY2hpbmUpLCBzb3VyY2U6ICRJTVBPUlQgfVxuICBGSU5BTDpcbiAgICByZXBsYWNlOiB7IGJ5OiBjcmVhdGVBY3RvciwgcmVwbGFjZTogaW50ZXJwcmV0LCBzb3VyY2U6ICRTVEVQMSB9XG5maXg6ICRGSU5BTFxuLS0tIFxucnVsZTogeyBwYXR0ZXJuOiAkTUFDSElORS53aXRoQ29uZmlnIH1cbmZpeDogJE1BQ0hJTkUucHJvdmlkZVxuLS0tXG5ydWxlOlxuICBraW5kOiBwcm9wZXJ0eV9pZGVudGlmaWVyXG4gIHJlZ2V4OiBec2VydmljZXMkXG4gIGluc2lkZTogeyBwYXR0ZXJuOiAgJE0ud2l0aENvbmZpZygkJCRBUkdTKSwgc3RvcEJ5OiBlbmQgfVxuZml4OiBhY3RvcnMiLCJzb3VyY2UiOiJpbXBvcnQgeyBNYWNoaW5lLCBpbnRlcnByZXQgfSBmcm9tICd4c3RhdGUnO1xuXG5jb25zdCBtYWNoaW5lID0gTWFjaGluZSh7IC8qLi4uKi99KTtcblxuY29uc3Qgc3BlY2lmaWNNYWNoaW5lID0gbWFjaGluZS53aXRoQ29uZmlnKHtcbiAgYWN0aW9uczogeyAvKiAuLi4gKi8gfSxcbiAgZ3VhcmRzOiB7IC8qIC4uLiAqLyB9LFxuICBzZXJ2aWNlczogeyAvKiAuLi4gKi8gfSxcbn0pO1xuXG5jb25zdCBhY3RvciA9IGludGVycHJldChzcGVjaWZpY01hY2hpbmUsIHtcbi8qIGFjdG9yIG9wdGlvbnMgKi9cbn0pOyJ9)

### Description

[XState](https://xstate.js.org/) is a state management/orchestration library based on state machines, statecharts, and the actor model. It allows you to model complex logic in event-driven ways, and orchestrate the behavior of many actors communicating with each other.

XState's v5 version introduced some breaking changes and new features compared to v4.
While the migration should be a straightforward process, it is a tedious process and requires knowledge of the differences between v4 and v5.

ast-grep provides a way to automate the process and a way to encode valuable knowledge to executable rules.

The following example picks up some migration items and demonstrates the power of ast-grep's rule system.

### YAML

The rules below correspond to XState v5's [`createMachine`](https://stately.ai/docs/migration#use-createmachine-not-machine), [`createActor`](https://stately.ai/docs/migration#use-createactor-not-interpret), and [`machine.provide`](https://stately.ai/docs/migration#use-machineprovide-not-machinewithconfig).

The example shows how ast-grep can use various features like [utility rule](/guide/rule-config/utility-rule.html), [transformation](/reference/yaml/transformation.html) and [multiple rule in single file](/reference/playground.html#test-multiple-rules) to automate the migration. Each rule has a clear and descriptive `id` field that explains its purpose.

For more information, you can use [Codemod AI](https://app.codemod.com/studio?ai_thread_id=new) to provide more detailed explanation for each rule.

```yaml
id: migrate-import-name
utils:
  FROM_XS: {kind: import_statement, has: {kind: string, regex: xstate}}
  XS_EXPORT:
    kind: identifier
    inside: { has: { matches: FROM_XS }, stopBy: end }
rule: { regex: ^Machine|interpret$, pattern: $IMPT, matches: XS_EXPORT }
transform:
  STEP1:
    replace: {by: create$1, replace: (Machine), source: $IMPT }
  FINAL:
    replace: { by: createActor, replace: interpret, source: $STEP1 }
fix: $FINAL

---

id: migrate-to-provide
rule: { pattern: $MACHINE.withConfig }
fix: $MACHINE.provide

---

id: migrate-to-actors
rule:
  kind: property_identifier
  regex: ^services$
  inside: { pattern:  $M.withConfig($$$ARGS), stopBy: end }
fix: actors
```

### Example

```js {1,3,5,8,11}
import { Machine, interpret } from 'xstate';

const machine = Machine({ /*...*/});

const specificMachine = machine.withConfig({
  actions: { /* ... */ },
  guards: { /* ... */ },
  services: { /* ... */ },
});

const actor = interpret(specificMachine, {
  /* actor options */
});
```

### Diff

```js
import { Machine, interpret } from 'xstate'; // [!code --]
import { createMachine, createActor } from 'xstate'; // [!code ++]

const machine = Machine({ /*...*/}); // [!code --]
const machine = createMachine({ /*...*/}); // [!code ++]

const specificMachine = machine.withConfig({ // [!code --]
const specificMachine = machine.provide({ // [!code ++]
  actions: { /* ... */ },
  guards: { /* ... */ },
  services: { /* ... */ }, // [!code --]
  actors: { /* ... */ }, // [!code ++]
});

const actor = interpret(specificMachine, { // [!code --]
const actor = createActor(specificMachine, { // [!code ++]
  /* actor options */
});
```

### Contributed by

Inspired by [XState's blog](https://stately.ai/blog/2023-12-01-xstate-v5).

## No `await` in `Promise.all` array&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiaWQ6IG5vLWF3YWl0LWluLXByb21pc2UtYWxsXG5zZXZlcml0eTogZXJyb3Jcbmxhbmd1YWdlOiBKYXZhU2NyaXB0XG5tZXNzYWdlOiBObyBhd2FpdCBpbiBQcm9taXNlLmFsbFxucnVsZTpcbiAgcGF0dGVybjogYXdhaXQgJEFcbiAgaW5zaWRlOlxuICAgIHBhdHRlcm46IFByb21pc2UuYWxsKCRfKVxuICAgIHN0b3BCeTpcbiAgICAgIG5vdDogeyBhbnk6IFt7a2luZDogYXJyYXl9LCB7a2luZDogYXJndW1lbnRzfV0gfVxuZml4OiAkQSIsInNvdXJjZSI6ImNvbnN0IFtmb28sIGJhcl0gPSBhd2FpdCBQcm9taXNlLmFsbChbXG4gIGF3YWl0IGdldEZvbygpLFxuICBnZXRCYXIoKSxcbiAgKGFzeW5jICgpID0+IHsgYXdhaXQgZ2V0QmF6KCl9KSgpLFxuXSkifQ==)

### Description

Using `await` inside an inline `Promise.all` array is usually a mistake, as it defeats the purpose of running the promises in parallel. Instead, the promises should be created without `await` and passed to `Promise.all`, which can then be awaited.

### YAML

```yaml
id: no-await-in-promise-all
language: typescript
rule:
  pattern: await $A
  inside:
    pattern: Promise.all($_)
    stopBy:
      not: { any: [{kind: array}, {kind: arguments}] }
fix: $A
```

### Example

```ts {2}
const [foo, bar] = await Promise.all([
  await getFoo(),
  getBar(),
  (async () => { await getBaz()})(),
])
```

### Diff

```ts
const [foo, bar] = await Promise.all([
  await getFoo(), // [!code --]
  getFoo(), // [!code ++]
  getBar(),
  (async () => { await getBaz()})(),
])
```

### Contributed by

Inspired by [Alvar Lagerlöf](https://twitter.com/alvarlagerlof)

## No `console` except in `catch` block&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImlmICRBLmhhc19mZWF0dXJlP1xuICAgICQkJEJcbmVsc2UgXG4gICAgJCQkQyBcbmVuZCAiLCJyZXdyaXRlIjoiJCQkQiIsImNvbmZpZyI6InJ1bGU6XG4gIGFueTpcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUuZXJyb3IoJCQkKVxuICAgICAgbm90OlxuICAgICAgICBpbnNpZGU6XG4gICAgICAgICAga2luZDogY2F0Y2hfY2xhdXNlXG4gICAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUuJE1FVEhPRCgkJCQpXG5jb25zdHJhaW50czpcbiAgTUVUSE9EOlxuICAgIHJlZ2V4OiAnbG9nfGRlYnVnfHdhcm4nXG5maXg6ICcnIiwic291cmNlIjoiY29uc29sZS5kZWJ1ZygnJylcbnRyeSB7XG4gICAgY29uc29sZS5sb2coJ2hlbGxvJylcbn0gY2F0Y2ggKGUpIHtcbiAgICBjb25zb2xlLmVycm9yKGUpXG59In0=)

### Description

Using `console` methods is usually for debugging purposes and therefore not suitable to ship to the client.
`console` can expose sensitive information, clutter the output, or affect the performance.

The only exception is using `console.error` to log errors in the catch block, which can be useful for debugging production.

### YAML

```yaml
id: no-console-except-error
language: typescript
rule:
  any:
    - pattern: console.error($$$)
      not:
        inside:
          kind: catch_clause
          stopBy: end
    - pattern: console.$METHOD($$$)
constraints:
  METHOD:
    regex: 'log|debug|warn'
```

### Example

```ts {1,3}
console.debug('')
try {
    console.log('hello')
} catch (e) {
    console.error(e) // OK
}
```

### Diff

```ts
console.debug('') // [!code --]
try {
    console.log('hello') // [!code --]
} catch (e) {
    console.error(e) // OK
}
```

### Contributed by

Inspired by [Jerry Mouse](https://github.com/WWK563388548)

## Find Import Usage

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTpcbiAgIyB0aGUgdXNhZ2VcbiAga2luZDogaWRlbnRpZmllclxuICBwYXR0ZXJuOiAkTU9EXG4gICMgaXRzIHJlbGF0aW9uc2hpcCB0byB0aGUgcm9vdFxuICBpbnNpZGU6XG4gICAgc3RvcEJ5OiBlbmRcbiAgICBraW5kOiBwcm9ncmFtXG4gICAgIyBhbmQgYmFjayBkb3duIHRvIHRoZSBpbXBvcnQgc3RhdGVtZW50XG4gICAgaGFzOlxuICAgICAga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgIyBhbmQgZGVlcGVyIGludG8gdGhlIGltcG9ydCBzdGF0ZW1lbnQgbG9va2luZyBmb3IgdGhlIG1hdGNoaW5nIGlkZW50aWZpZXJcbiAgICAgIGhhczpcbiAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAgICAga2luZDogaW1wb3J0X3NwZWNpZmllclxuICAgICAgICBwYXR0ZXJuOiAkTU9EICMgc2FtZSBwYXR0ZXJuIGFzIHRoZSB1c2FnZSBpcyBlbmZvcmNlZCBoZXJlIiwic291cmNlIjoiaW1wb3J0IHsgTW9uZ29DbGllbnQgfSBmcm9tICdtb25nb2RiJztcbmNvbnN0IHVybCA9ICdtb25nb2RiOi8vbG9jYWxob3N0OjI3MDE3JztcbmFzeW5jIGZ1bmN0aW9uIHJ1bigpIHtcbiAgY29uc3QgY2xpZW50ID0gbmV3IE1vbmdvQ2xpZW50KHVybCk7XG59XG4ifQ==)

### Description

It is common to find the usage of an imported module in a codebase. This rule helps you to find the usage of an imported module in your codebase.
The idea of this rule can be broken into several parts:

* Find the use of an identifier `$MOD`
* To find the import, we first need to find the root file of which `$MOD` is  `inside`
* The `program` file `has` an `import` statement
* The `import` statement `has` the identifier `$MOD`

### YAML

```yaml
id: find-import-usage
language: typescript
rule:
  kind: identifier # ast-grep requires a kind
  pattern: $MOD   # the identifier to find
  inside: # find the root
    stopBy: end
    kind: program
    has: # and has the import statement
      kind: import_statement
      has: # look for the matching identifier
        stopBy: end
        kind: import_specifier
        pattern: $MOD # same pattern as the usage is enforced here
```

### Example

```ts {4}
import { MongoClient } from 'mongodb';
const url = 'mongodb://localhost:27017';
async function run() {
  const client = new MongoClient(url);
}
```

### Contributed by

[Steven Love](https://github.com/StevenLove)

## Switch Chai from `should` style to `expect`&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IHNob3VsZF90b19leHBlY3RfaW5zdGFuY2VvZlxubGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbnJ1bGU6XG4gIGFueTpcbiAgLSBwYXR0ZXJuOiAkTkFNRS5zaG91bGQuYmUuYW4uaW5zdGFuY2VvZigkVFlQRSlcbiAgLSBwYXR0ZXJuOiAkTkFNRS5zaG91bGQuYmUuYW4uaW5zdGFuY2VPZigkVFlQRSlcbmZpeDogfC1cbiAgZXhwZWN0KCROQU1FKS5pbnN0YW5jZU9mKCRUWVBFKVxuLS0tXG5pZDogc2hvdWxkX3RvX2V4cGVjdF9nZW5lcmljU2hvdWxkQmVcbmxhbmd1YWdlOiBUeXBlU2NyaXB0XG5ydWxlOlxuICBwYXR0ZXJuOiAkTkFNRS5zaG91bGQuYmUuJFBST1BcbmZpeDogfC1cbiAgZXhwZWN0KCROQU1FKS50by5iZS4kUFJPUFxuIiwic291cmNlIjoiaXQoJ3Nob3VsZCBwcm9kdWNlIGFuIGluc3RhbmNlIG9mIGNob2tpZGFyLkZTV2F0Y2hlcicsICgpID0+IHtcbiAgd2F0Y2hlci5zaG91bGQuYmUuYW4uaW5zdGFuY2VvZihjaG9raWRhci5GU1dhdGNoZXIpO1xufSk7XG5pdCgnc2hvdWxkIGV4cG9zZSBwdWJsaWMgQVBJIG1ldGhvZHMnLCAoKSA9PiB7XG4gIHdhdGNoZXIub24uc2hvdWxkLmJlLmEoJ2Z1bmN0aW9uJyk7XG4gIHdhdGNoZXIuZW1pdC5zaG91bGQuYmUuYSgnZnVuY3Rpb24nKTtcbiAgd2F0Y2hlci5hZGQuc2hvdWxkLmJlLmEoJ2Z1bmN0aW9uJyk7XG4gIHdhdGNoZXIuY2xvc2Uuc2hvdWxkLmJlLmEoJ2Z1bmN0aW9uJyk7XG4gIHdhdGNoZXIuZ2V0V2F0Y2hlZC5zaG91bGQuYmUuYSgnZnVuY3Rpb24nKTtcbn0pOyJ9)

### Description

[Chai](https://www.chaijs.com) is a BDD / TDD assertion library for JavaScript. It comes with [two styles](https://www.chaijs.com/) of assertions: `should` and `expect`.

The `expect` interface provides a function as a starting point for chaining your language assertions and works with `undefined` and `null` values.
The `should` style allows for the same chainable assertions as the expect interface, however it extends each object with a should property to start your chain and [does not work](https://www.chaijs.com/guide/styles/#should-extras) with `undefined` and `null` values.

This rule migrates Chai `should` style assertions to `expect` style assertions. Note this is an example rule and a excerpt from [the original rules](https://github.com/43081j/codemods/blob/cddfe101e7f759e4da08b7e2f7bfe892c20f6f48/codemods/chai-should-to-expect.yml).

### YAML

```yaml
id: should_to_expect_instanceof
language: TypeScript
rule:
  any:
  - pattern: $NAME.should.be.an.instanceof($TYPE)
  - pattern: $NAME.should.be.an.instanceOf($TYPE)
fix: |-
  expect($NAME).instanceOf($TYPE)
---
id: should_to_expect_genericShouldBe
language: TypeScript
rule:
  pattern: $NAME.should.be.$PROP
fix: |-
  expect($NAME).to.be.$PROP
```

### Example

```js {2,5-9}
it('should produce an instance of chokidar.FSWatcher', () => {
  watcher.should.be.an.instanceof(chokidar.FSWatcher);
});
it('should expose public API methods', () => {
  watcher.on.should.be.a('function');
  watcher.emit.should.be.a('function');
  watcher.add.should.be.a('function');
  watcher.close.should.be.a('function');
  watcher.getWatched.should.be.a('function');
});
```

### Diff

```js
it('should produce an instance of chokidar.FSWatcher', () => {
  watcher.should.be.an.instanceof(chokidar.FSWatcher); // [!code --]
  expect(watcher).instanceOf(chokidar.FSWatcher); // [!code ++]
});
it('should expose public API methods', () => {
  watcher.on.should.be.a('function');   // [!code --]
  watcher.emit.should.be.a('function'); // [!code --]
  watcher.add.should.be.a('function');  // [!code --]
  watcher.close.should.be.a('function'); // [!code --]
  watcher.getWatched.should.be.a('function'); // [!code --]
  expect(watcher.on).to.be.a('function'); // [!code ++]
  expect(watcher.emit).to.be.a('function'); // [!code ++]
  expect(watcher.add).to.be.a('function'); // [!code ++]
  expect(watcher.close).to.be.a('function'); // [!code ++]
  expect(watcher.getWatched).to.be.a('function'); // [!code ++]
});
```

### Contributed by

[James](https://bsky.app/profile/43081j.com), by [this post](https://bsky.app/profile/43081j.com/post/3lgimzfxza22i)

### Exercise

Exercise left to the reader: can you write a rule to implement [this migration to `node:assert`](https://github.com/paulmillr/chokidar/pull/1409/files)?

## Speed up Barrel Import&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBpbXBvcnQgeyQkJElERU5UU30gZnJvbSAnLi9iYXJyZWwnXG5yZXdyaXRlcnM6XG4tIGlkOiByZXdyaXRlLWlkZW50aWZlclxuICBydWxlOlxuICAgIHBhdHRlcm46ICRJREVOVFxuICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgZml4OiBpbXBvcnQgJElERU5UIGZyb20gJy4vYmFycmVsLyRJREVOVCdcbnRyYW5zZm9ybTpcbiAgSU1QT1JUUzpcbiAgICByZXdyaXRlOlxuICAgICAgcmV3cml0ZXJzOiBbcmV3cml0ZS1pZGVudGlmZXJdXG4gICAgICBzb3VyY2U6ICQkJElERU5UU1xuICAgICAgam9pbkJ5OiBcIlxcblwiXG5maXg6ICRJTVBPUlRTIiwic291cmNlIjoiaW1wb3J0IHsgYSwgYiwgYyB9IGZyb20gJy4vYmFycmVsJzsifQ==)

### Description

A [barrel import](https://adrianfaciu.dev/posts/barrel-files/) is a way to consolidate the exports of multiple modules into a single convenient module that can be imported using a single import statement. For instance, `import {a, b, c} from './barrel'`.

It has [some](https://vercel.com/blog/how-we-optimized-package-imports-in-next-js) [benefits](https://marvinh.dev/blog/speeding-up-javascript-ecosystem-part-7/) to import each module directly from its own file without going through the barrel file.
Such as reducing [bundle size](https://dev.to/tassiofront/barrel-files-and-why-you-should-stop-using-them-now-bc4), improving building time or avoiding [conflicting names](https://flaming.codes/posts/barrel-files-in-javascript/).

### YAML

```yaml
id: speed-up-barrel-import
language: typescript
# find the barrel import statement
rule:
  pattern: import {$$$IDENTS} from './barrel'
# rewrite imported identifiers to direct imports
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  fix: import $IDENT from './barrel/$IDENT'
# apply the rewriter to the import statement
transform:
  IMPORTS:
    rewrite:
      rewriters: [rewrite-identifer]
      # $$$IDENTS contains imported identifiers
      source: $$$IDENTS
      # join the rewritten imports by newline
      joinBy: "\n"
fix: $IMPORTS
```

### Example

```ts {1}
import {a, b, c} from './barrel'
```

### Diff

```ts
import {a, b, c} from './barrel' // [!code --]
import a from './barrel/a' // [!code ++]
import b from './barrel/b' // [!code ++]
import c from './barrel/c' // [!code ++]
```

### Contributed by

[Herrington Darkholme](https://x.com/hd_nvim)

## Missing Component Decorator

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImltcG9ydCAkQSBmcm9tICdhbmltZWpzJyIsInJld3JpdGUiOiJpbXBvcnQgeyBhbmltZSBhcyAkQSB9IGZyb20gJ2FuaW1lJyIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IG1pc3NpbmctY29tcG9uZW50LWRlY29yYXRvclxubWVzc2FnZTogWW91J3JlIHVzaW5nIGFuIEFuZ3VsYXIgbGlmZWN5Y2xlIG1ldGhvZCwgYnV0IG1pc3NpbmcgYW4gQW5ndWxhciBAQ29tcG9uZW50KCkgZGVjb3JhdG9yLlxubGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbnNldmVyaXR5OiB3YXJuaW5nXG5ydWxlOlxuICBwYXR0ZXJuOlxuICAgIGNvbnRleHQ6ICdjbGFzcyBIaSB7ICRNRVRIT0QoKSB7ICQkJF99IH0nXG4gICAgc2VsZWN0b3I6IG1ldGhvZF9kZWZpbml0aW9uXG4gIGluc2lkZTpcbiAgICBwYXR0ZXJuOiAnY2xhc3MgJEtMQVNTICQkJF8geyAkJCRfIH0nXG4gICAgc3RvcEJ5OiBlbmRcbiAgICBub3Q6XG4gICAgICBoYXM6XG4gICAgICAgIHBhdHRlcm46ICdAQ29tcG9uZW50KCQkJF8pJ1xuY29uc3RyYWludHM6XG4gIE1FVEhPRDpcbiAgICByZWdleDogbmdPbkluaXR8bmdPbkRlc3Ryb3lcbmxhYmVsczpcbiAgS0xBU1M6XG4gICAgc3R5bGU6IHByaW1hcnlcbiAgICBtZXNzYWdlOiBcIlRoaXMgY2xhc3MgaXMgbWlzc2luZyB0aGUgZGVjb3JhdG9yLlwiXG4gIE1FVEhPRDpcbiAgICBzdHlsZTogc2Vjb25kYXJ5XG4gICAgbWVzc2FnZTogXCJUaGlzIGlzIGFuIEFuZ3VsYXIgbGlmZWN5Y2xlIG1ldGhvZC5cIlxubWV0YWRhdGE6XG4gIGNvbnRyaWJ1dGVkQnk6IHNhbXdpZ2h0dCIsInNvdXJjZSI6ImNsYXNzIE5vdENvbXBvbmVudCB7XG4gICAgbmdPbkluaXQoKSB7fVxufVxuXG5AQ29tcG9uZW50KClcbmNsYXNzIEtsYXNzIHtcbiAgICBuZ09uSW5pdCgpIHt9XG59In0=)

### Description

Angular lifecycle methods are a set of methods that allow you to hook into the lifecycle of an Angular component or directive.
They must be used within a class that is decorated with the `@Component()` decorator.

### YAML

This rule illustrates how to use custom labels to highlight specific parts of the code.

```yaml
id: missing-component-decorator
message: You're using an Angular lifecycle method, but missing an Angular @Component() decorator.
language: TypeScript
severity: warning
rule:
  pattern:
    context: 'class Hi { $METHOD() { $$$_} }'
    selector: method_definition
  inside:
    pattern: 'class $KLASS $$$_ { $$$_ }'
    stopBy: end
    not:
      has:
        pattern: '@Component($$$_)'
constraints:
  METHOD:
    regex: ngOnInit|ngOnDestroy
labels:
  KLASS:
    style: primary
    message: "This class is missing the decorator."
  METHOD:
    style: secondary
    message: "This is an Angular lifecycle method."
metadata:
  contributedBy: samwightt
```

### Example

```ts {2}
class NotComponent {
    ngOnInit() {}
}

@Component()
class Klass {
    ngOnInit() {}
}
```

### Contributed by

[Sam Wight](https://github.com/samwightt).

## Find Import Identifiers

* [Playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiIjIGZpbmQtYWxsLWltcG9ydHMtYW5kLXJlcXVpcmVzLnlhbWxcbmlkOiBmaW5kLWFsbC1pbXBvcnRzLWFuZC1yZXF1aXJlc1xubGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbm1lc3NhZ2U6IEZvdW5kIG1vZHVsZSBpbXBvcnQgb3IgcmVxdWlyZS5cbnNldmVyaXR5OiBpbmZvXG5ydWxlOlxuICBhbnk6XG4gICAgIyBBTElBUyBJTVBPUlRTXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGltcG9ydCB7IE9SSUdJTkFMIGFzIEFMSUFTIH0gZnJvbSAnU09VUkNFJ1xuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgICMgMS4gVGFyZ2V0IHRoZSBzcGVjaWZpYyBub2RlIHR5cGUgZm9yIG5hbWVkIGltcG9ydHNcbiAgICAgICAgLSBraW5kOiBpbXBvcnRfc3BlY2lmaWVyXG4gICAgICAgICMgMi4gRW5zdXJlIGl0ICpoYXMqIGFuICdhbGlhcycgZmllbGQsIGNhcHR1cmluZyB0aGUgYWxpYXMgaWRlbnRpZmllclxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBhbGlhc1xuICAgICAgICAgICAgcGF0dGVybjogJEFMSUFTXG4gICAgICAgICMgMy4gQ2FwdHVyZSB0aGUgb3JpZ2luYWwgaWRlbnRpZmllciAod2hpY2ggaGFzIHRoZSAnbmFtZScgZmllbGQpXG4gICAgICAgIC0gaGFzOlxuICAgICAgICAgICAgZmllbGQ6IG5hbWVcbiAgICAgICAgICAgIHBhdHRlcm46ICRPUklHSU5BTFxuICAgICAgICAjIDQuIEZpbmQgYW4gQU5DRVNUT1IgaW1wb3J0X3N0YXRlbWVudCBhbmQgY2FwdHVyZSBpdHMgc291cmNlIHBhdGhcbiAgICAgICAgLSBpbnNpZGU6XG4gICAgICAgICAgICBzdG9wQnk6IGVuZCAjIDw8PC0tLSBUaGlzIGlzIHRoZSBrZXkgZml4ISBTZWFyY2ggYW5jZXN0b3JzLlxuICAgICAgICAgICAga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgICAgICAgaGFzOiAjIEVuc3VyZSB0aGUgZm91bmQgaW1wb3J0X3N0YXRlbWVudCBoYXMgdGhlIHNvdXJjZSBmaWVsZFxuICAgICAgICAgICAgICBmaWVsZDogc291cmNlXG4gICAgICAgICAgICAgIHBhdHRlcm46ICRTT1VSQ0VcblxuICAgICMgREVGQVVMVCBJTVBPUlRTXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGltcG9ydCB7IE9SSUdJTkFMIH0gZnJvbSAnU09VUkNFJ1xuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgIC0ga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgICMgRW5zdXJlIGl0IGhhcyBhbiBpbXBvcnRfY2xhdXNlLi4uXG4gICAgICAgICAgICBraW5kOiBpbXBvcnRfY2xhdXNlXG4gICAgICAgICAgICBoYXM6XG4gICAgICAgICAgICAgICMgLi4udGhhdCBkaXJlY3RseSBjb250YWlucyBhbiBpZGVudGlmaWVyICh0aGUgZGVmYXVsdCBpbXBvcnQgbmFtZSlcbiAgICAgICAgICAgICAgIyBUaGlzIGlkZW50aWZpZXIgaXMgTk9UIHVuZGVyIGEgJ25hbWVkX2ltcG9ydHMnIG9yICduYW1lc3BhY2VfaW1wb3J0JyBub2RlXG4gICAgICAgICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgICAgICAgcGF0dGVybjogJERFRkFVTFRfTkFNRVxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBzb3VyY2VcbiAgICAgICAgICAgIHBhdHRlcm46ICRTT1VSQ0VcbiAgICBcbiAgICAjIFJFR1VMQVIgSU1QT1JUU1xuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgIyBpbXBvcnQgeyBPUklHSU5BTCB9IGZyb20gJ1NPVVJDRSdcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgIC0gYWxsOlxuICAgICAgICAjIDEuIFRhcmdldCB0aGUgc3BlY2lmaWMgbm9kZSB0eXBlIGZvciBuYW1lZCBpbXBvcnRzXG4gICAgICAgIC0ga2luZDogaW1wb3J0X3NwZWNpZmllclxuICAgICAgICAjIDIuIEVuc3VyZSBpdCAqaGFzKiBhbiAnYWxpYXMnIGZpZWxkLCBjYXB0dXJpbmcgdGhlIGFsaWFzIGlkZW50aWZpZXJcbiAgICAgICAgLSBoYXM6XG4gICAgICAgICAgICBmaWVsZDogbmFtZVxuICAgICAgICAgICAgcGF0dGVybjogJE9SSUdJTkFMXG4gICAgICAgICMgNC4gRmluZCBhbiBBTkNFU1RPUiBpbXBvcnRfc3RhdGVtZW50IGFuZCBjYXB0dXJlIGl0cyBzb3VyY2UgcGF0aFxuICAgICAgICAtIGluc2lkZTpcbiAgICAgICAgICAgIHN0b3BCeTogZW5kICMgPDw8LS0tIFRoaXMgaXMgdGhlIGtleSBmaXghIFNlYXJjaCBhbmNlc3RvcnMuXG4gICAgICAgICAgICBraW5kOiBpbXBvcnRfc3RhdGVtZW50XG4gICAgICAgICAgICBoYXM6ICMgRW5zdXJlIHRoZSBmb3VuZCBpbXBvcnRfc3RhdGVtZW50IGhhcyB0aGUgc291cmNlIGZpZWxkXG4gICAgICAgICAgICAgIGZpZWxkOiBzb3VyY2VcbiAgICAgICAgICAgICAgcGF0dGVybjogJFNPVVJDRVxuXG4gICAgIyBEWU5BTUlDIElNUE9SVFMgKFNpbmdsZSBWYXJpYWJsZSBBc3NpZ25tZW50KSBcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgICMgZWc6IChjb25zdCBWQVJfTkFNRSA9IHJlcXVpcmUoJ1NPVVJDRScpKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgIC0ga2luZDogdmFyaWFibGVfZGVjbGFyYXRvclxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBuYW1lXG4gICAgICAgICAgICBraW5kOiBpZGVudGlmaWVyXG4gICAgICAgICAgICBwYXR0ZXJuOiAkVkFSX05BTUUgIyBDYXB0dXJlIHRoZSBzaW5nbGUgdmFyaWFibGUgbmFtZVxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiB2YWx1ZVxuICAgICAgICAgICAgYW55OlxuICAgICAgICAgICAgICAjIERpcmVjdCBjYWxsXG4gICAgICAgICAgICAgIC0gYWxsOiAjIFdyYXAgY29uZGl0aW9ucyBpbiBhbGxcbiAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogZnVuY3Rpb24sIHJlZ2V4OiAnXihyZXF1aXJlfGltcG9ydCkkJyB9XG4gICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogYXJndW1lbnRzLCBoYXM6IHsga2luZDogc3RyaW5nLCBwYXR0ZXJuOiAkU09VUkNFIH0gfSAjIENhcHR1cmUgc291cmNlXG4gICAgICAgICAgICAgICMgQXdhaXRlZCBjYWxsXG4gICAgICAgICAgICAgIC0ga2luZDogYXdhaXRfZXhwcmVzc2lvblxuICAgICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICAgIGFsbDogIyBXcmFwIGNvbmRpdGlvbnMgaW4gYWxsXG4gICAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgIC0gaGFzOiB7IGZpZWxkOiBmdW5jdGlvbiwgcmVnZXg6ICdeKHJlcXVpcmV8aW1wb3J0KSQnIH1cbiAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGFyZ3VtZW50cywgaGFzOiB7IGtpbmQ6IHN0cmluZywgcGF0dGVybjogJFNPVVJDRSB9IH0gIyBDYXB0dXJlIHNvdXJjZVxuXG4gICAgIyBEWU5BTUlDIElNUE9SVFMgKERlc3RydWN0dXJlZCBTaG9ydGhhbmQgQXNzaWdubWVudCkgICAgIFxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgIyBlZzogKGNvbnN0IHsgT1JJR0lOQUwgfSA9IHJlcXVpcmUoJ1NPVVJDRScpKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgICMgMS4gVGFyZ2V0IHRoZSBzaG9ydGhhbmQgaWRlbnRpZmllciB3aXRoaW4gdGhlIHBhdHRlcm5cbiAgICAgICAgLSBraW5kOiBzaG9ydGhhbmRfcHJvcGVydHlfaWRlbnRpZmllcl9wYXR0ZXJuXG4gICAgICAgIC0gcGF0dGVybjogJE9SSUdJTkFMXG4gICAgICAgICMgMi4gRW5zdXJlIGl0J3MgaW5zaWRlIGFuIG9iamVjdF9wYXR0ZXJuIHRoYXQgaXMgdGhlIG5hbWUgb2YgYSB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgIC0gaW5zaWRlOlxuICAgICAgICAgICAga2luZDogb2JqZWN0X3BhdHRlcm5cbiAgICAgICAgICAgIGluc2lkZTogIyBDaGVjayB0aGUgdmFyaWFibGVfZGVjbGFyYXRvciBpdCBiZWxvbmdzIHRvXG4gICAgICAgICAgICAgIGtpbmQ6IHZhcmlhYmxlX2RlY2xhcmF0b3JcbiAgICAgICAgICAgICAgIyAzLiBDaGVjayB0aGUgdmFsdWUgYXNzaWduZWQgYnkgdGhlIHZhcmlhYmxlX2RlY2xhcmF0b3JcbiAgICAgICAgICAgICAgaGFzOlxuICAgICAgICAgICAgICAgIGZpZWxkOiB2YWx1ZVxuICAgICAgICAgICAgICAgIGFueTpcbiAgICAgICAgICAgICAgICAgICMgRGlyZWN0IGNhbGxcbiAgICAgICAgICAgICAgICAgIC0gYWxsOlxuICAgICAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGZ1bmN0aW9uLCByZWdleDogJ14ocmVxdWlyZXxpbXBvcnQpJCcgfVxuICAgICAgICAgICAgICAgICAgICAgIC0gaGFzOiB7IGZpZWxkOiBhcmd1bWVudHMsIGhhczogeyBraW5kOiBzdHJpbmcsIHBhdHRlcm46ICRTT1VSQ0UgfSB9ICMgQ2FwdHVyZSBzb3VyY2VcbiAgICAgICAgICAgICAgICAgICMgQXdhaXRlZCBjYWxsXG4gICAgICAgICAgICAgICAgICAtIGtpbmQ6IGF3YWl0X2V4cHJlc3Npb25cbiAgICAgICAgICAgICAgICAgICAgaGFzOlxuICAgICAgICAgICAgICAgICAgICAgIGFsbDpcbiAgICAgICAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogZnVuY3Rpb24sIHJlZ2V4OiAnXihyZXF1aXJlfGltcG9ydCkkJyB9XG4gICAgICAgICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogYXJndW1lbnRzLCBoYXM6IHsga2luZDogc3RyaW5nLCBwYXR0ZXJuOiAkU09VUkNFIH0gfSAjIENhcHR1cmUgc291cmNlXG4gICAgICAgICAgICAgIHN0b3BCeTogZW5kICMgU2VhcmNoIGFuY2VzdG9ycyB0byBmaW5kIHRoZSBjb3JyZWN0IHZhcmlhYmxlX2RlY2xhcmF0b3JcblxuICAgICMgRFlOQU1JQyBJTVBPUlRTIChEZXN0cnVjdHVyZWQgQWxpYXMgQXNzaWdubWVudCkgXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGVnOiAoY29uc3QgeyBPUklHSU5BTDogQUxJQVMgfSA9IHJlcXVpcmUoJ1NPVVJDRScpKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgICMgMS4gVGFyZ2V0IHRoZSBwYWlyX3BhdHRlcm4gZm9yIGFsaWFzZWQgZGVzdHJ1Y3R1cmluZ1xuICAgICAgICAtIGtpbmQ6IHBhaXJfcGF0dGVyblxuICAgICAgICAjIDIuIENhcHR1cmUgdGhlIG9yaWdpbmFsIGlkZW50aWZpZXIgKGtleSlcbiAgICAgICAgLSBoYXM6XG4gICAgICAgICAgICBmaWVsZDoga2V5XG4gICAgICAgICAgICBraW5kOiBwcm9wZXJ0eV9pZGVudGlmaWVyICMgQ291bGQgYmUgc3RyaW5nL251bWJlciBsaXRlcmFsIHRvbywgYnV0IHByb3BlcnR5X2lkZW50aWZpZXIgaXMgY29tbW9uXG4gICAgICAgICAgICBwYXR0ZXJuOiAkT1JJR0lOQUxcbiAgICAgICAgIyAzLiBDYXB0dXJlIHRoZSBhbGlhcyBpZGVudGlmaWVyICh2YWx1ZSlcbiAgICAgICAgLSBoYXM6XG4gICAgICAgICAgICBmaWVsZDogdmFsdWVcbiAgICAgICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgICAgIHBhdHRlcm46ICRBTElBU1xuICAgICAgICAjIDQuIEVuc3VyZSBpdCdzIGluc2lkZSBhbiBvYmplY3RfcGF0dGVybiB0aGF0IGlzIHRoZSBuYW1lIG9mIGEgdmFyaWFibGVfZGVjbGFyYXRvclxuICAgICAgICAtIGluc2lkZTpcbiAgICAgICAgICAgIGtpbmQ6IG9iamVjdF9wYXR0ZXJuXG4gICAgICAgICAgICBpbnNpZGU6ICMgQ2hlY2sgdGhlIHZhcmlhYmxlX2RlY2xhcmF0b3IgaXQgYmVsb25ncyB0b1xuICAgICAgICAgICAgICBraW5kOiB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgICAgICAgICMgNS4gQ2hlY2sgdGhlIHZhbHVlIGFzc2lnbmVkIGJ5IHRoZSB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICBmaWVsZDogdmFsdWVcbiAgICAgICAgICAgICAgICBhbnk6XG4gICAgICAgICAgICAgICAgICAjIERpcmVjdCBjYWxsXG4gICAgICAgICAgICAgICAgICAtIGFsbDpcbiAgICAgICAgICAgICAgICAgICAgICAtIGtpbmQ6IGNhbGxfZXhwcmVzc2lvblxuICAgICAgICAgICAgICAgICAgICAgIC0gaGFzOiB7IGZpZWxkOiBmdW5jdGlvbiwgcmVnZXg6ICdeKHJlcXVpcmV8aW1wb3J0KSQnIH1cbiAgICAgICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogYXJndW1lbnRzLCBoYXM6IHsga2luZDogc3RyaW5nLCBwYXR0ZXJuOiAkU09VUkNFIH0gfSAjIENhcHR1cmUgc291cmNlXG4gICAgICAgICAgICAgICAgICAjIEF3YWl0ZWQgY2FsbFxuICAgICAgICAgICAgICAgICAgLSBraW5kOiBhd2FpdF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICAgICAgICBhbGw6XG4gICAgICAgICAgICAgICAgICAgICAgICAtIGtpbmQ6IGNhbGxfZXhwcmVzc2lvblxuICAgICAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGZ1bmN0aW9uLCByZWdleDogJ14ocmVxdWlyZXxpbXBvcnQpJCcgfVxuICAgICAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGFyZ3VtZW50cywgaGFzOiB7IGtpbmQ6IHN0cmluZywgcGF0dGVybjogJFNPVVJDRSB9IH0gIyBDYXB0dXJlIHNvdXJjZVxuICAgICAgICAgICAgICBzdG9wQnk6IGVuZCAjIFNlYXJjaCBhbmNlc3RvcnMgdG8gZmluZCB0aGUgY29ycmVjdCB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgICAgICBzdG9wQnk6IGVuZCAjIEVuc3VyZSB3ZSBjaGVjayBhbmNlc3RvcnMgZm9yIHRoZSB2YXJpYWJsZV9kZWNsYXJhdG9yXG5cbiAgICAjIERZTkFNSUMgSU1QT1JUUyAoU2lkZSBFZmZlY3QgLyBTb3VyY2UgT25seSkgXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGVnOiAocmVxdWlyZSgnU09VUkNFJykpXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAtIGFsbDpcbiAgICAgICAgLSBraW5kOiBzdHJpbmcgIyBUYXJnZXQgdGhlIHNvdXJjZSBzdHJpbmcgbGl0ZXJhbCBkaXJlY3RseVxuICAgICAgICAtIHBhdHRlcm46ICRTT1VSQ0VcbiAgICAgICAgLSBpbnNpZGU6ICMgU3RyaW5nIG11c3QgYmUgdGhlIGFyZ3VtZW50IG9mIHJlcXVpcmUoKSBvciBpbXBvcnQoKVxuICAgICAgICAgICAga2luZDogYXJndW1lbnRzXG4gICAgICAgICAgICBwYXJlbnQ6XG4gICAgICAgICAgICAgIGtpbmQ6IGNhbGxfZXhwcmVzc2lvblxuICAgICAgICAgICAgICBoYXM6XG4gICAgICAgICAgICAgICAgZmllbGQ6IGZ1bmN0aW9uXG4gICAgICAgICAgICAgICAgIyBNYXRjaCAncmVxdWlyZScgaWRlbnRpZmllciBvciAnaW1wb3J0JyBrZXl3b3JkIHVzZWQgZHluYW1pY2FsbHlcbiAgICAgICAgICAgICAgICByZWdleDogJ14ocmVxdWlyZXxpbXBvcnQpJCdcbiAgICAgICAgICAgIHN0b3BCeTogZW5kICMgU2VhcmNoIGFuY2VzdG9ycyBpZiBuZWVkZWQgKGZvciB0aGUgYXJndW1lbnRzL2NhbGxfZXhwcmVzc2lvbilcbiAgICAgICAgLSBub3Q6XG4gICAgICAgICAgICBpbnNpZGU6XG4gICAgICAgICAgICAgIGtpbmQ6IGxleGljYWxfZGVjbGFyYXRpb25cbiAgICAgICAgICAgICAgc3RvcEJ5OiBlbmQgIyBTZWFyY2ggYWxsIGFuY2VzdG9ycyB1cCB0byB0aGUgcm9vdFxuXG4gICAgIyBOQU1FU1BBQ0UgSU1QT1JUUyBcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgICMgZWc6IChpbXBvcnQgKiBhcyBucyBmcm9tICdtb2QnKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgIC0ga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGtpbmQ6IGltcG9ydF9jbGF1c2VcbiAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAga2luZDogbmFtZXNwYWNlX2ltcG9ydFxuICAgICAgICAgICAgICBoYXM6XG4gICAgICAgICAgICAgICAgIyBuYW1lc3BhY2VfaW1wb3J0J3MgY2hpbGQgaWRlbnRpZmllciBpcyB0aGUgYWxpYXNcbiAgICAgICAgICAgICAgICBraW5kOiBpZGVudGlmaWVyXG4gICAgICAgICAgICAgICAgcGF0dGVybjogJE5BTUVTUEFDRV9BTElBU1xuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBzb3VyY2VcbiAgICAgICAgICAgIHBhdHRlcm46ICRTT1VSQ0VcblxuICAgICMgU0lERSBFRkZFQ1QgSU1QT1JUUyBcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgICMgZWc6IChpbXBvcnQgJ21vZCcpXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAtIGFsbDpcbiAgICAgICAgLSBraW5kOiBpbXBvcnRfc3RhdGVtZW50XG4gICAgICAgIC0gbm90OiAjIE11c3QgTk9UIGhhdmUgYW4gaW1wb3J0X2NsYXVzZVxuICAgICAgICAgICAgaGFzOiB7IGtpbmQ6IGltcG9ydF9jbGF1c2UgfVxuICAgICAgICAtIGhhczogIyBCdXQgbXVzdCBoYXZlIGEgc291cmNlXG4gICAgICAgICAgICBmaWVsZDogc291cmNlXG4gICAgICAgICAgICBwYXR0ZXJuOiAkU09VUkNFXG4iLCJzb3VyY2UiOiIvL0B0cy1ub2NoZWNrXG4vLyBOYW1lZCBpbXBvcnRcbmltcG9ydCB7IHRlc3RpbmcgfSBmcm9tICcuL3Rlc3RzJztcblxuLy8gQWxpYXNlZCBpbXBvcnRcbmltcG9ydCB7IHRlc3RpbmcgYXMgdGVzdCB9IGZyb20gJy4vdGVzdHMyJztcblxuLy8gRGVmYXVsdCBpbXBvcnRcbmltcG9ydCBoZWxsbyBmcm9tICdoZWxsb193b3JsZDEnO1xuXG4vLyBOYW1lc3BhY2UgaW1wb3J0XG5pbXBvcnQgKiBhcyBzb21ldGhpbmcgZnJvbSAnaGVsbG9fd29ybGQyJztcblxuLy8gU2lkZS1lZmZlY3QgaW1wb3J0XG5pbXBvcnQgJ0BmYXN0aWZ5L3N0YXRpYyc7XG5cbi8vIFR5cGUgaW1wb3J0XG5pbXBvcnQge3R5cGUgaGVsbG8xMjQzIGFzIHRlc3Rpbmd9IGZyb20gJ2hlbGxvJztcblxuLy8gUmVxdWlyZSBwYXR0ZXJuc1xuY29uc3QgbW9kID0gcmVxdWlyZSgnc29tZS1tb2R1bGUnKTtcbnJlcXVpcmUoJ3BvbHlmaWxsJyk7XG5cbi8vIERlc3RydWN0dXJlZCByZXF1aXJlXG5jb25zdCB7IHRlc3QxMjIsIHRlc3QyIH0gPSByZXF1aXJlKCcuL2Rlc3RydWN0dXJlZDEnKTtcbi8vIEFsaWFzZWQgcmVxdWlyZVxuY29uc3QgeyB0ZXN0MTIyOiB0ZXN0MTIzLCB0ZXN0MjogdGVzdDIzLCB0ZXN0MzogdGVzdDMzIH0gPSByZXF1aXJlKCcuL2Rlc3RydWN0dXJlZDInKTtcblxuLy8gTWl4ZWQgaW1wb3J0c1xuaW1wb3J0IGRlZmF1bHRFeHBvcnQsIHsgbmFtZWRFeHBvcnQgfSBmcm9tICcuL21peGVkJztcbmltcG9ydCBkZWZhdWx0RXhwb3J0MiwgKiBhcyBuYW1lc3BhY2UgZnJvbSAnLi9taXhlZDInO1xuXG5cbi8vIE11bHRpcGxlIGltcG9ydCBsaW5lcyBmcm9tIHRoZSBzYW1lIGZpbGVcbmltcG9ydCB7IG9uZSwgdHdvIGFzIGFsaWFzLCB0aHJlZSB9IGZyb20gJy4vbXVsdGlwbGUnO1xuaW1wb3J0IHsgbmV2ZXIsIGdvbm5hLCBnaXZlLCB5b3UsIHVwIH0gZnJvbSAnLi9tdWx0aXBsZSc7XG5cbi8vIFN0cmluZyBsaXRlcmFsIHZhcmlhdGlvbnNcbmltcG9ydCB7IHRlc3QxIH0gZnJvbSBcIi4vZG91YmxlLXF1b3RlZFwiO1xuaW1wb3J0IHsgdGVzdDIgfSBmcm9tICcuL3NpbmdsZS1xdW90ZWQnO1xuXG4vLyBNdWx0aWxpbmUgaW1wb3J0c1xuaW1wb3J0IHtcbiAgICBsb25nSW1wb3J0MSxcbiAgICBsb25nSW1wb3J0MiBhcyBhbGlhczIsXG4gICAgbG9uZ0ltcG9ydDNcbn0gZnJvbSAnLi9tdWx0aWxpbmUnO1xuXG4vLyBEeW5hbWljIGltcG9ydHNcbmNvbnN0IGR5bmFtaWNNb2R1bGUgPSBpbXBvcnQoJy4vZHluYW1pYzEnKTtcbmNvbnN0IHt0ZXN0aW5nLCB0ZXN0aW5nMTIzfSA9IGltcG9ydCgnLi9keW5hbWljMicpO1xuY29uc3QgYXN5bmNEeW5hbWljTW9kdWxlID0gYXdhaXQgaW1wb3J0KCcuL2FzeW5jX2R5bmFtaWMxJykudGhlbihtb2R1bGUgPT4gbW9kdWxlLmRlZmF1bHQpO1xuLy8gQWxpYXNlZCBkeW5hbWljIGltcG9ydFxuY29uc3QgeyBvcmlnaW5hbElkZW50aWZpZXI6IGFsaWFzZWREeW5hbWljSW1wb3J0fSA9IGF3YWl0IGltcG9ydCgnLi9hc3luY19keW5hbWljMicpO1xuXG4vLyBDb21tZW50cyBpbiBpbXBvcnRzXG5pbXBvcnQgLyogdGVzdCAqLyB7IFxuICAgIC8vIENvbW1lbnQgaW4gaW1wb3J0XG4gICAgY29tbWVudGVkSW1wb3J0IFxufSBmcm9tICcuL2NvbW1lbnRlZCc7IC8vIEVuZCBvZiBsaW5lIGNvbW1lbnQgXG5cblxuIn0=)

### Description

Finding import metadata can be useful. Below is a comprehensive snippet for extracting identifiers from various import statements:

* Alias Imports (`import { hello as world } from './file'`)
* Default & Regular Imports (`import test from './my-test`')
* Dynamic Imports (`require(...)`, and `import(...)`)
* Side Effect & Namespace Imports (`import * as myCode from './code`')

### YAML

```yaml
# find-all-imports-and-identifiers.yaml
id: find-all-imports-and-identifiers
language: TypeScript
rule:
  any:
    # ALIAS IMPORTS
    # ------------------------------------------------------------
    # import { ORIGINAL as ALIAS } from 'SOURCE'
    # ------------------------------------------------------------
    - all:
        # 1. Target the specific node type for named imports
        - kind: import_specifier
        # 2. Ensure it *has* an 'alias' field, capturing the alias identifier
        - has:
            field: alias
            pattern: $ALIAS
        # 3. Capture the original identifier (which has the 'name' field)
        - has:
            field: name
            pattern: $ORIGINAL
        # 4. Find an ANCESTOR import_statement and capture its source path
        - inside:
            stopBy: end # <<<--- Search ancestors.
            kind: import_statement
            has: # Ensure the found import_statement has the source field
              field: source
              pattern: $SOURCE

    # DEFAULT IMPORTS
    # ------------------------------------------------------------
    # import { ORIGINAL } from 'SOURCE'
    # ------------------------------------------------------------
    - all:
        - kind: import_statement
        - has:
            # Ensure it has an import_clause...
            kind: import_clause
            has:
              # ...that directly contains an identifier (the default import name)
              # This identifier is NOT under a 'named_imports' or 'namespace_import' node
              kind: identifier
              pattern: $DEFAULT_NAME
        - has:
            field: source
            pattern: $SOURCE

    # REGULAR IMPORTS
    # ------------------------------------------------------------
    # import { ORIGINAL } from 'SOURCE'
    # ------------------------------------------------------------
    - all:
        # 1. Target the specific node type for named imports
        - kind: import_specifier
        # 2. Ensure it *has* an 'alias' field, capturing the alias identifier
        - has:
            field: name
            pattern: $ORIGINAL
        # 4. Find an ANCESTOR import_statement and capture its source path
        - inside:
            stopBy: end # <<<--- This is the key fix! Search ancestors.
            kind: import_statement
            has: # Ensure the found import_statement has the source field
              field: source
              pattern: $SOURCE

    # DYNAMIC IMPORTS (Single Variable Assignment)
    # ------------------------------------------------------------
    # const VAR_NAME = require('SOURCE')
    # ------------------------------------------------------------
    - all:
        - kind: variable_declarator
        - has:
            field: name
            kind: identifier
            pattern: $VAR_NAME # Capture the single variable name
        - has:
            field: value
            any:
              # Direct call
              - all: # Wrap conditions in all
                  - kind: call_expression
                  - has: { field: function, regex: '^(require|import)$' }
                  - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
              # Awaited call
              - kind: await_expression
                has:
                  all: # Wrap conditions in all
                    - kind: call_expression
                    - has: { field: function, regex: '^(require|import)$' }
                    - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source

    # DYNAMIC IMPORTS (Destructured Shorthand Assignment)
    # ------------------------------------------------------------
    # const { ORIGINAL } = require('SOURCE')
    # ------------------------------------------------------------
    - all:
        # 1. Target the shorthand identifier within the pattern
        - kind: shorthand_property_identifier_pattern
        - pattern: $ORIGINAL
        # 2. Ensure it's inside an object_pattern that is the name of a variable_declarator
        - inside:
            kind: object_pattern
            inside: # Check the variable_declarator it belongs to
              kind: variable_declarator
              # 3. Check the value assigned by the variable_declarator
              has:
                field: value
                any:
                  # Direct call
                  - all:
                      - kind: call_expression
                      - has: { field: function, regex: '^(require|import)$' }
                      - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
                  # Awaited call
                  - kind: await_expression
                    has:
                      all:
                        - kind: call_expression
                        - has: { field: function, regex: '^(require|import)$' }
                        - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
              stopBy: end # Search ancestors to find the correct variable_declarator

    # DYNAMIC IMPORTS (Destructured Alias Assignment)
    # ------------------------------------------------------------
    # const { ORIGINAL: ALIAS } = require('SOURCE')
    # ------------------------------------------------------------
    - all:
        # 1. Target the pair_pattern for aliased destructuring
        - kind: pair_pattern
        # 2. Capture the original identifier (key)
        - has:
            field: key
            kind: property_identifier # Could be string/number literal too, but property_identifier is common
            pattern: $ORIGINAL
        # 3. Capture the alias identifier (value)
        - has:
            field: value
            kind: identifier
            pattern: $ALIAS
        # 4. Ensure it's inside an object_pattern that is the name of a variable_declarator
        - inside:
            kind: object_pattern
            inside: # Check the variable_declarator it belongs to
              kind: variable_declarator
              # 5. Check the value assigned by the variable_declarator
              has:
                field: value
                any:
                  # Direct call
                  - all:
                      - kind: call_expression
                      - has: { field: function, regex: '^(require|import)$' }
                      - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
                  # Awaited call
                  - kind: await_expression
                    has:
                      all:
                        - kind: call_expression
                        - has: { field: function, regex: '^(require|import)$' }
                        - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
              stopBy: end # Search ancestors to find the correct variable_declarator
            stopBy: end # Ensure we check ancestors for the variable_declarator

    # DYNAMIC IMPORTS (Side Effect / Source Only)
    # ------------------------------------------------------------
    # require('SOURCE')
    # ------------------------------------------------------------
    - all:
        - kind: string # Target the source string literal directly
        - pattern: $SOURCE
        - inside: # String must be the argument of require() or import()
            kind: arguments
            parent:
              kind: call_expression
              has:
                field: function
                # Match 'require' identifier or 'import' keyword used dynamically
                regex: '^(require|import)$'
            stopBy: end # Search ancestors if needed (for the arguments/call_expression)
        - not:
            inside:
              kind: lexical_declaration
              stopBy: end # Search all ancestors up to the root

    # NAMESPACE IMPORTS
    # ------------------------------------------------------------
    # import * as ns from 'mod'
    # ------------------------------------------------------------
    - all:
        - kind: import_statement
        - has:
            kind: import_clause
            has:
              kind: namespace_import
              has:
                # namespace_import's child identifier is the alias
                kind: identifier
                pattern: $NAMESPACE_ALIAS
        - has:
            field: source
            pattern: $SOURCE

    # SIDE EFFECT IMPORTS
    # ------------------------------------------------------------
    # import 'mod'
    # ------------------------------------------------------------
    - all:
        - kind: import_statement
        - not: # Must NOT have an import_clause
            has: { kind: import_clause }
        - has: # But must have a source
            field: source
            pattern: $SOURCE
```

### Example

```ts {60}
//@ts-nocheck
// Named import
import { testing } from './tests';

// Aliased import
import { testing as test } from './tests2';

// Default import
import hello from 'hello_world1';

// Namespace import
import * as something from 'hello_world2';

// Side-effect import
import '@fastify/static';

// Type import
import {type hello1243 as testing} from 'hello';

// Require patterns
const mod = require('some-module');
require('polyfill');

// Destructured require
const { test122, test2 } = require('./destructured1');
// Aliased require
const { test122: test123, test2: test23, test3: test33 } = require('./destructured2');

// Mixed imports
import defaultExport, { namedExport } from './mixed';
import defaultExport2, * as namespace from './mixed2';


// Multiple import lines from the same file
import { one, two as alias, three } from './multiple';
import { never, gonna, give, you, up } from './multiple';

// String literal variations
import { test1 } from "./double-quoted";
import { test2 } from './single-quoted';

// Multiline imports
import {
    longImport1,
    longImport2 as alias2,
    longImport3
} from './multiline';

// Dynamic imports
const dynamicModule = import('./dynamic1');
const {testing, testing123} = import('./dynamic2');
const asyncDynamicModule = await import('./async_dynamic1').then(module => module.default);
// Aliased dynamic import
const { originalIdentifier: aliasedDynamicImport} = await import('./async_dynamic2');

// Comments in imports
import /* test */ {
    // Comment in import
    commentedImport
} from './commented'; // End of line comment
```

### Contributed by

[Michael Angelo Rivera](https://github.com/michaelangeloio)

## Use Logical Assignment Operators

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiJEEgPSAkQSB8fCAkQiIsInJld3JpdGUiOiIkQSB8fD0gJEIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiMgWUFNTCBSdWxlIGlzIG1vcmUgcG93ZXJmdWwhXG4jIGh0dHBzOi8vYXN0LWdyZXAuZ2l0aHViLmlvL2d1aWRlL3J1bGUtY29uZmlnLmh0bWwjcnVsZVxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogY29uc29sZS5sb2coJEEpXG4gICAgLSBwYXR0ZXJuOiBjb25zb2xlLmRlYnVnKCRBKVxuZml4OlxuICBsb2dnZXIubG9nKCRBKSIsInNvdXJjZSI6ImNvbnN0IGEgPSB7IGR1cmF0aW9uOiA1MCwgdGl0bGU6ICcnIH07XG5cbmEuZHVyYXRpb24gPSBhLmR1cmF0aW9uIHx8IDEwO1xuY29uc29sZS5sb2coYS5kdXJhdGlvbik7XG5hLnRpdGxlID0gYS50aXRsZSB8fCAndGl0bGUgaXMgZW1wdHkuJztcbmNvbnNvbGUubG9nKGEudGl0bGUpO1xuIn0=)

### Description

A logical assignment operator in JavaScript combines a logical operation (like OR or nullish coalescing) with an assignment. It updates a variable or property only under specific conditions, making code more concise.

This is a relatively new feature in JavaScript (introduced in ES2021), so older codebases might not use it yet. This rule identifies instances where a variable is assigned a value using a logical OR (`||`) operation and suggests replacing it with the more concise logical assignment operator (`||=`).

### Pattern

```shell
ast-grep -p '$A = $A || $B' -r '$A ||= $B' -l ts
```

### Example

```ts {3,5}
const a = { duration: 50, title: '' };

a.duration = a.duration || 10;
console.log(a.duration);
a.title = a.title || 'title is empty.';
console.log(a.title);
```

### Diff

```ts
const a = { duration: 50, title: '' };

a.duration = a.duration || 10; // [!code --]
a.duration ||= 10; // [!code ++]
console.log(a.duration);
a.title = a.title || 'title is empty.'; // [!code --]
a.title ||= 'title is empty.'; // [!code ++]
console.log(a.title);
```

### Contributed by

Inspired by [this tweet](https://x.com/YTCodeAntonio/status/1973720331656605809).

---

---
url: /catalog/tsx.md
---
# TSX

This page curates a list of example ast-grep rules to check and to rewrite TypeScript with JSX syntax.

:::danger TSX and TypeScript are different.
TSX differs from TypeScript because it is an extension of the latter that supports JSX elements.
They need distinct parsers because of [conflicting syntax](https://www.typescriptlang.org/docs/handbook/jsx.html#the-as-operator).

In order to reduce rule duplication, you can use the [`languageGlobs`](/reference/sgconfig.html#languageglobs) option to force ast-grep to use parse `.ts` files as TSX.
:::

## Unnecessary `useState` Type&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoidHlwZXNjcmlwdCIsInF1ZXJ5IjoidXNlU3RhdGU8c3RyaW5nPigkQSkiLCJyZXdyaXRlIjoidXNlU3RhdGUoJEEpIiwiY29uZmlnIjoiIyBZQU1MIFJ1bGUgaXMgbW9yZSBwb3dlcmZ1bCFcbiMgaHR0cHM6Ly9hc3QtZ3JlcC5naXRodWIuaW8vZ3VpZGUvcnVsZS1jb25maWcuaHRtbCNydWxlXG5ydWxlOlxuICBhbnk6XG4gICAgLSBwYXR0ZXJuOiBjb25zb2xlLmxvZygkQSlcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUuZGVidWcoJEEpXG5maXg6XG4gIGxvZ2dlci5sb2coJEEpIiwic291cmNlIjoiZnVuY3Rpb24gQ29tcG9uZW50KCkge1xuICBjb25zdCBbbmFtZSwgc2V0TmFtZV0gPSB1c2VTdGF0ZTxzdHJpbmc+KCdSZWFjdCcpXG59In0=)

### Description

React's [`useState`](https://react.dev/reference/react/useState) is a Hook that lets you add a state variable to your component. The type annotation of `useState`'s generic type argument, for example `useState<number>(123)`, is unnecessary if TypeScript can infer the type of the state variable from the initial value.

We can usually skip annotating if the generic type argument is a single primitive type like `number`, `string` or `boolean`.

### Pattern

::: code-group

```bash [number]
ast-grep -p 'useState<number>($A)' -r 'useState($A)' -l tsx
```

```bash [string]
ast-grep -p 'useState<string>($A)' -r 'useState($A)'
```

```bash [boolean]
ast-grep -p 'useState<boolean>($A)' -r 'useState($A)'
```

:::

### Example

```ts {2}
function Component() {
  const [name, setName] = useState<string>('React')
}
```

### Diff

```ts
function Component() {
  const [name, setName] = useState<string>('React') // [!code --]
  const [name, setName] = useState('React') // [!code ++]
}
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim)

## Avoid `&&` short circuit in JSX&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiY29uc29sZS5sb2coJE1BVENIKSIsInJld3JpdGUiOiJsb2dnZXIubG9nKCRNQVRDSCkiLCJjb25maWciOiJpZDogZG8td2hhdC1icm9vb29vb2tseW4tc2FpZFxubGFuZ3VhZ2U6IFRzeFxuc2V2ZXJpdHk6IGVycm9yXG5ydWxlOlxuICBraW5kOiBqc3hfZXhwcmVzc2lvblxuICBoYXM6XG4gICAgcGF0dGVybjogJEEgJiYgJEJcbiAgbm90OlxuICAgIGluc2lkZTpcbiAgICAgIGtpbmQ6IGpzeF9hdHRyaWJ1dGVcbmZpeDogXCJ7JEEgPyAkQiA6IG51bGx9XCIiLCJzb3VyY2UiOiI8ZGl2PntcbiAgbnVtICYmIDxkaXYvPlxufTwvZGl2PiJ9)

### Description

In [React](https://react.dev/learn/conditional-rendering), you can conditionally render JSX using JavaScript syntax like `if` statements, `&&`, and `? :` operators.
However, you should almost never put numbers on the left side of `&&`. This is because React will render the number `0`, instead of the JSX element on the right side. A concrete example will be conditionally rendering a list when the list is not empty.

This rule will find and fix any short-circuit rendering in JSX and rewrite it to a ternary operator.

### YAML

```yaml
id: do-what-brooooooklyn-said
language: Tsx
rule:
  kind: jsx_expression
  has:
    pattern: $A && $B
  not:
    inside:
      kind: jsx_attribute
fix: "{$A ? $B : null}"
```

### Example

```tsx {1}
<div>{ list.length && list.map(i => <p/>) }</div>
```

### Diff

```tsx
<div>{ list.length && list.map(i => <p/>) }</div> // [!code --]
<div>{ list.length ?  list.map(i => <p/>) : null }</div> // [!code ++]
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [@Brooooook\_lyn](https://twitter.com/Brooooook_lyn/status/1666637274757595141)

## Rewrite MobX Component Style&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoicnVsZTpcbiAgcGF0dGVybjogZXhwb3J0IGNvbnN0ICRDT01QID0gb2JzZXJ2ZXIoJEZVTkMpXG5maXg6IHwtXG4gIGNvbnN0IEJhc2UkQ09NUCA9ICRGVU5DXG4gIGV4cG9ydCBjb25zdCAkQ09NUCA9IG9ic2VydmVyKEJhc2UkQ09NUCkiLCJzb3VyY2UiOiJleHBvcnQgY29uc3QgRXhhbXBsZSA9IG9ic2VydmVyKCgpID0+IHtcbiAgcmV0dXJuIDxkaXY+SGVsbG8gV29ybGQ8L2Rpdj5cbn0pIn0=)

### Description

React and MobX are libraries that help us build user interfaces with JavaScript.

[React hooks](https://react.dev/reference/react) allow us to use state and lifecycle methods in functional components. But we need follow some hook rules, or React may break. [MobX](https://mobx.js.org/react-integration.html) has an `observer` function that makes a component update when data changes.

When we use the `observer` function like this:

```JavaScript
export const Example = observer(() => {…})
```

ESLint, the tool that checks hooks, thinks that `Example` is not a React component, but just a regular function. So it does not check the hooks inside it, and we may miss some wrong usages.

To fix this, we need to change our component style to this:

```JavaScript
const BaseExample = () => {…}
const Example = observer(BaseExample)
```

Now ESLint can see that `BaseExample` is a React component, and it can check the hooks inside it.

### YAML

```yaml
id: rewrite-mobx-component
language: typescript
rule:
  pattern: export const $COMP = observer($FUNC)
fix: |-
  const Base$COMP = $FUNC
  export const $COMP = observer(Base$COMP)
```

### Example

```js {1-3}
export const Example = observer(() => {
  return <div>Hello World</div>
})
```

### Diff

```js
export const Example = observer(() => { // [!code --]
  return <div>Hello World</div>         // [!code --]
})                                      // [!code --]
const BaseExample = () => {             // [!code ++]
  return <div>Hello World</div>         // [!code ++]
}                                       // [!code ++]
export const Example = observer(BaseExample) // [!code ++]
```

### Contributed by

[Bryan Lee](https://twitter.com/meetliby/status/1698601672568901723)

## Avoid Unnecessary React Hook

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InV0aWxzOlxuICBob29rX2NhbGw6XG4gICAgaGFzOlxuICAgICAga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICByZWdleDogXnVzZVxuICAgICAgc3RvcEJ5OiBlbmRcbnJ1bGU6XG4gIGFueTpcbiAgLSBwYXR0ZXJuOiBmdW5jdGlvbiAkRlVOQygkJCQpIHsgJCQkIH1cbiAgLSBwYXR0ZXJuOiBsZXQgJEZVTkMgPSAoJCQkKSA9PiAkJCQgXG4gIC0gcGF0dGVybjogY29uc3QgJEZVTkMgPSAoJCQkKSA9PiAkJCRcbiAgaGFzOlxuICAgIHBhdHRlcm46ICRCT0RZXG4gICAga2luZDogc3RhdGVtZW50X2Jsb2NrXG4gICAgc3RvcEJ5OiBlbmQgXG5jb25zdHJhaW50czpcbiAgRlVOQzoge3JlZ2V4OiBedXNlIH1cbiAgQk9EWTogeyBub3Q6IHsgbWF0Y2hlczogaG9va19jYWxsIH0gfSBcbiIsInNvdXJjZSI6ImZ1bmN0aW9uIHVzZUlBbU5vdEhvb2tBY3R1YWxseShhcmdzKSB7XG4gICAgY29uc29sZS5sb2coJ0NhbGxlZCBpbiBSZWFjdCBidXQgSSBkb250IG5lZWQgdG8gYmUgYSBob29rJylcbiAgICByZXR1cm4gYXJncy5sZW5ndGhcbn1cbmNvbnN0IHVzZUlBbU5vdEhvb2tUb28gPSAoLi4uYXJncykgPT4ge1xuICAgIGNvbnNvbGUubG9nKCdDYWxsZWQgaW4gUmVhY3QgYnV0IEkgZG9udCBuZWVkIHRvIGJlIGEgaG9vaycpXG4gICAgcmV0dXJuIGFyZ3MubGVuZ3RoXG59XG5cbmZ1bmN0aW9uIHVzZUhvb2soKSB7XG4gICAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICAgIGNvbnNvbGUubG9nKCdSZWFsIGhvb2snKSAgIFxuICAgIH0pXG59In0=)

### Description

React hook is a powerful feature in React that allows you to use state and other React features in a functional component.

However, you should avoid using hooks when you don't need them. If the code does not contain using any other React hooks,
it can be rewritten to a plain function. This can help to separate your application logic from the React-specific UI logic.

### YAML

```yaml
id: unnecessary-react-hook
language: Tsx
utils:
  hook_call:
    has:
      kind: call_expression
      regex: ^use
      stopBy: end
rule:
  any:
  - pattern: function $FUNC($$$) { $$$ }
  - pattern: let $FUNC = ($$$) => $$$
  - pattern: const $FUNC = ($$$) => $$$
  has:
    pattern: $BODY
    kind: statement_block
    stopBy: end
constraints:
  FUNC: {regex: ^use }
  BODY: { not: { matches: hook_call } }
```

### Example

```tsx {1-8}
function useIAmNotHookActually(args) {
    console.log('Called in React but I dont need to be a hook')
    return args.length
}
const useIAmNotHookToo = (...args) => {
    console.log('Called in React but I dont need to be a hook')
    return args.length
}

function useTrueHook() {
    useEffect(() => {
      console.log('Real hook')
    })
}
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim)

## Reverse React Compiler™&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogcmV3cml0ZS1jYWNoZSBcbmxhbmd1YWdlOiB0c3hcbnJ1bGU6XG4gIGFueTpcbiAgLSBwYXR0ZXJuOiB1c2VDYWxsYmFjaygkRk4sICQkJClcbiAgLSBwYXR0ZXJuOiBtZW1vKCRGTiwgJCQkKVxuZml4OiAkRk5cblxuLS0tXG5cbmlkOiByZXdyaXRlLXVzZS1tZW1vXG5sYW5ndWFnZTogdHN4XG5ydWxlOiB7IHBhdHRlcm46ICd1c2VNZW1vKCRGTiwgJCQkKScgfVxuZml4OiAoJEZOKSgpIiwic291cmNlIjoiY29uc3QgQ29tcG9uZW50ID0gKCkgPT4ge1xuICBjb25zdCBbY291bnQsIHNldENvdW50XSA9IHVzZVN0YXRlKDApXG4gIGNvbnN0IGluY3JlbWVudCA9IHVzZUNhbGxiYWNrKCgpID0+IHtcbiAgICBzZXRDb3VudCgocHJldkNvdW50KSA9PiBwcmV2Q291bnQgKyAxKVxuICB9LCBbXSlcbiAgY29uc3QgZXhwZW5zaXZlQ2FsY3VsYXRpb24gPSB1c2VNZW1vKCgpID0+IHtcbiAgICAvLyBtb2NrIEV4cGVuc2l2ZSBjYWxjdWxhdGlvblxuICAgIHJldHVybiBjb3VudCAqIDJcbiAgfSwgW2NvdW50XSlcblxuICByZXR1cm4gKFxuICAgIDw+XG4gICAgICA8cD5FeHBlbnNpdmUgUmVzdWx0OiB7ZXhwZW5zaXZlQ2FsY3VsYXRpb259PC9wPlxuICAgICAgPGJ1dHRvbiBvbkNsaWNrPXtpbmNyZW1lbnR9Pntjb3VudH08L2J1dHRvbj5cbiAgICA8Lz5cbiAgKVxufSJ9)

### Description

React Compiler is a build-time only tool that automatically optimizes your React app, working with plain JavaScript and understanding the Rules of React without requiring a rewrite. It optimizes apps by automatically memoizing code, similar to `useMemo`, `useCallback`, and `React.memo`, reducing unnecessary recomputation due to incorrect or forgotten memoization.

Reverse React Compiler™ is a [parody tweet](https://x.com/aidenybai/status/1881397529369034997) that works in the opposite direction. It takes React code and removes memoization,  guaranteed to make your code slower. ([not](https://x.com/kentcdodds/status/1881404373646880997) [necessarily](https://dev.to/prathamisonline/are-you-over-using-usememo-and-usecallback-hooks-in-react-5lp))

It is originally written in Babel and this is an [ast-grep version](https://x.com/hd_nvim/status/1881402678493970620) of it.

:::details The Original Babel Implementation
For comparison purposes only. Note the original code [does not correctly rewrite](https://x.com/hd_nvim/status/1881404893136896415) `useMemo`.

```js
const ReverseReactCompiler = ({ types: t }) => ({
  visitor: {
    CallExpression(path) {
      const callee = path.node.callee;
      if (
        t.isIdentifier(callee, { name: "useMemo" }) ||
        t.isIdentifier(callee, { name: "useCallback" }) ||
        t.isIdentifier(callee, { name: "memo" })
      ) {
        path.replaceWith(args[0]);
      }
    },
  },
});
```

:::

### YAML

```yaml
id: rewrite-cache
language: tsx
rule:
  any:
  - pattern: useCallback($FN, $$$)
  - pattern: memo($FN, $$$)
fix: $FN
---
id: rewrite-use-memo
language: tsx
rule: { pattern: 'useMemo($FN, $$$)' }
fix: ($FN)()   # need IIFE to wrap memo function
```

### Example

```tsx {3-5,6-9}
const Component = () => {
  const [count, setCount] = useState(0)
  const increment = useCallback(() => {
    setCount((prevCount) => prevCount + 1)
  }, [])
  const expensiveCalculation = useMemo(() => {
    // mock Expensive calculation
    return count * 2
  }, [count])

  return (
    <>
      <p>Expensive Result: {expensiveCalculation}</p>
      <button onClick={increment}>{count}</button>
    </>
  )
}
```

### Diff

```tsx
const Component = () => {
  const [count, setCount] = useState(0)
  const increment = useCallback(() => {     // [!code --]
    setCount((prevCount) => prevCount + 1)  // [!code --]
  }, [])                                 // [!code --]
  const increment = () => {         // [!code ++]
    setCount((prevCount) => prevCount + 1) // [!code ++]
  } // [!code ++]
  const expensiveCalculation = useMemo(() => { // [!code --]
    // mock Expensive calculation             // [!code --]
    return count * 2                        // [!code --]
  }, [count])                             // [!code --]
  const expensiveCalculation = (() => { // [!code ++]
    // mock Expensive calculation      // [!code ++]
    return count * 2                 // [!code ++]
  })()                            // [!code ++]
  return (
    <>
      <p>Expensive Result: {expensiveCalculation}</p>
      <button onClick={increment}>{count}</button>
    </>
  )
}
```

### Contributed by

Inspired by [Aiden Bai](https://twitter.com/aidenybai)

## Avoid nested links

* [Playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiaWYgKCRBKSB7ICQkJEIgfSIsInJld3JpdGUiOiJpZiAoISgkQSkpIHtcbiAgICByZXR1cm47XG59XG4kJCRCIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogbm8tbmVzdGVkLWxpbmtzXG5sYW5ndWFnZTogdHN4XG5zZXZlcml0eTogZXJyb3JcbnJ1bGU6XG4gIHBhdHRlcm46IDxhICQkJD4kJCRBPC9hPlxuICBoYXM6XG4gICAgcGF0dGVybjogPGEgJCQkPiQkJDwvYT5cbiAgICBzdG9wQnk6IGVuZCIsInNvdXJjZSI6ImZ1bmN0aW9uIENvbXBvbmVudCgpIHtcbiAgcmV0dXJuIDxhIGhyZWY9Jy9kZXN0aW5hdGlvbic+XG4gICAgPGEgaHJlZj0nL2Fub3RoZXJkZXN0aW5hdGlvbic+TmVzdGVkIGxpbmshPC9hPlxuICA8L2E+O1xufVxuZnVuY3Rpb24gT2theUNvbXBvbmVudCgpIHtcbiAgcmV0dXJuIDxhIGhyZWY9Jy9kZXN0aW5hdGlvbic+XG4gICAgSSBhbSBqdXN0IGEgbGluay5cbiAgPC9hPjtcbn0ifQ==)

### Description

React will produce a warning message if you nest a link element inside of another link element. This rule will catch this mistake!

### YAML

```yaml
id: no-nested-links
language: tsx
severity: error
rule:
  pattern: <a $$$>$$$A</a>
  has:
    pattern: <a $$$>$$$</a>
    stopBy: end
```

### Example

```tsx {1-5}
function Component() {
  return <a href='/destination'>
    <a href='/anotherdestination'>Nested link!</a>
  </a>;
}
function OkayComponent() {
  return <a href='/destination'>
    I am just a link.
  </a>;
}
```

### Contributed by

[Tom MacWright](https://macwright.com/)

## Rename SVG Attribute&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogcmV3cml0ZS1zdmctYXR0cmlidXRlXG5sYW5ndWFnZTogdHN4XG5ydWxlOlxuICBwYXR0ZXJuOiAkUFJPUFxuICByZWdleDogKFthLXpdKyktKFthLXpdKVxuICBraW5kOiBwcm9wZXJ0eV9pZGVudGlmaWVyXG4gIGluc2lkZTpcbiAgICBraW5kOiBqc3hfYXR0cmlidXRlXG50cmFuc2Zvcm06XG4gIE5FV19QUk9QOlxuICAgIGNvbnZlcnQ6XG4gICAgICBzb3VyY2U6ICRQUk9QXG4gICAgICB0b0Nhc2U6IGNhbWVsQ2FzZVxuZml4OiAkTkVXX1BST1AiLCJzb3VyY2UiOiJjb25zdCBlbGVtZW50ID0gKFxuICA8c3ZnIHdpZHRoPVwiMTAwXCIgaGVpZ2h0PVwiMTAwXCIgdmlld0JveD1cIjAgMCAxMDAgMTAwXCI+XG4gICAgPHBhdGggZD1cIk0xMCAyMCBMMzAgNDBcIiBzdHJva2UtbGluZWNhcD1cInJvdW5kXCIgZmlsbC1vcGFjaXR5PVwiMC41XCIgLz5cbiAgPC9zdmc+XG4pIn0=)

### Description

[SVG](https://en.wikipedia.org/wiki/SVG)(Scalable Vector Graphics)s' hyphenated names are not compatible with JSX syntax in React. JSX requires [camelCase naming](https://react.dev/learn/writing-markup-with-jsx#3-camelcase-salls-most-of-the-things) for attributes.
For example, an SVG attribute like `stroke-linecap` needs to be renamed to `strokeLinecap` to work correctly in React.

### YAML

```yaml
id: rewrite-svg-attribute
language: tsx
rule:
  pattern: $PROP            # capture in metavar
  regex: ([a-z]+)-([a-z])   # hyphenated name
  kind: property_identifier
  inside:
    kind: jsx_attribute     # in JSX attribute
transform:
  NEW_PROP:                 # new property name
    convert:                # use ast-grep's convert
      source: $PROP
      toCase: camelCase     # to camelCase naming
fix: $NEW_PROP
```

### Example

```tsx {3}
const element = (
  <svg width="100" height="100" viewBox="0 0 100 100">
    <path d="M10 20 L30 40" stroke-linecap="round" fill-opacity="0.5" />
  </svg>
)
```

### Diff

```ts
const element = (
  <svg width="100" height="100" viewBox="0 0 100 100">
    <path d="M10 20 L30 40" stroke-linecap="round" fill-opacity="0.5" /> // [!code --]
    <path d="M10 20 L30 40" strokeLinecap="round" fillOpacity="0.5" />   // [!code ++]
  </svg>
)
```

### Contributed by

Inspired by [SVG Renamer](https://admondtamang.medium.com/introducing-svg-renamer-your-solution-for-react-svg-attributes-26503382d5a8)

---

---
url: /catalog/yaml.md
---
# YAML

This page curates a list of example ast-grep rules to check and to rewrite YAML code.

## Find key/value and Show Message

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InlhbWwiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6ImlkOiBkZXRlY3QtaG9zdC1wb3J0XG5tZXNzYWdlOiBZb3UgYXJlIHVzaW5nICRIT1NUIG9uIFBvcnQgJFBPUlQsIHBsZWFzZSBjaGFuZ2UgaXQgdG8gODAwMFxuc2V2ZXJpdHk6IGVycm9yXG5ydWxlOlxuICBhbnk6XG4gIC0gcGF0dGVybjogfFxuICAgICBwb3J0OiAkUE9SVFxuICAtIHBhdHRlcm46IHxcbiAgICAgaG9zdDogJEhPU1QiLCJzb3VyY2UiOiJkYjpcbiAgIHVzZXJuYW1lOiByb290XG4gICBwYXNzd29yZDogcm9vdFxuXG5zZXJ2ZXI6XG4gIGhvc3Q6IDEyNy4wLjAuMVxuICBwb3J0OiA4MDAxIn0=)

### Description

This YAML rule helps detecting specific host and port configurations in your code. For example, it checks if the port is set to something other than 8000 or if a particular host is used. It provides an error message prompting you to update the configuration.

### YAML

```yaml
id: detect-host-port
message: You are using $HOST on Port $PORT, please change it to 8000
severity: error
rule:
  any:
    - pattern: |
        port: $PORT
    - pattern: |
        host: $HOST
```

### Example

```yaml {5,6}
db:
  username: root
  password: root
server:
  host: 127.0.0.1
  port: 8001
```

### Contributed by

[rohitcoder](https://twitter.com/rohitcoder) on [Discord](https://discord.com/invite/4YZjf6htSQ).

---

---
url: /cheatsheet/rule.md
---
# Rule Cheat Sheet

This cheat sheet provides a concise overview of ast-grep's rule object configuration, covering Atomic, Relational, and Composite rules, along with notes on Utility rules. It's designed as a handy reference for common usage.

## Atomic Rules Cheat Sheet

These are your precision tools, matching individual AST nodes based on their inherent properties.

```yaml
pattern: console.log($ARG)
```

🧩 Match a node by code structure. e.g. `console.log` call with a single `$ARG`

```yaml
pattern:
  context: '{ key: value }'
  selector: pair
```

🧩 To parse ambiguous patterns, use `context` and specify `selector` AST to search.

```yaml
kind: if_statement
```

🏷️ Match an AST node by its `kind` name


```yaml
regex: ^regex.+$
```

🔍 Matches node text content against a [Rust regular expression](https://docs.rs/regex/latest/regex/)

```yaml
nthChild: 1
```

🔢 Find a node by its **1-based index** among its *named siblings*

```yaml
nthChild:
  position: 2
  reverse: true
  ofRule: { kind: argument_list }
```

🔢 Advanced positional control: `position`, `reverse` (count from end), or filter siblings using `ofRule`

```yaml
range:
  start: { line: 0, column: 0 }
  end: { line: 0, column: 13 }
```

🎯 Matches a node based on its character span: 0-based, inclusive start, exclusive end

## Relational Rules Cheat Sheet

These powerful rules define how nodes relate to each other structurally. Think of them as your AST GPS!

```yaml
inside:
  kind: function_declaration
```

🏠 Target node must appear **inside** its *parent/ancestor* node matching the sub-rule

```yaml
has:
  kind: method_definition
```

🌳 Target node must **have** a *child/descendant* node matching the sub-rule

```yaml
has:
  kind: statement_block
  field: body
```

🌳 `field` makes `has`/`inside` match nodes by their [semantic role](/advanced/core-concepts.html#kind-vs-field)

```yaml
precedes:
  pattern: function $FUNC() { $$ }
```

◀️ Target node must appear *before* another node matching the sub-rule

```yaml
follows:
  pattern: let x = 10;
```

▶️ Target node must appear *after* another node matching the sub-rule.

```yaml
inside:
  kind: function_declaration
  stopBy: end
```

🏠 `stopBy` makes relational rules search all the way to the end, not just immediate neighbors.

## Composite Rules Cheat Sheet

Combine multiple rules using Boolean logic. Crucially, these operations apply to a single target node!

```yaml
all:
  - pattern: const $VAR = $VALUE
  - has: { kind: string_literal }
```

✅ Node must satisfy **ALL** the rules in the list.

```yaml
any:
  - pattern: let $X = $Y
  - pattern: const $X = $Y
```

🧡 Node must satisfy **AT LEAST ONE** of the rules in the list.

```yaml
not:
  pattern: console.log($$)
```

🚫 Node must **NOT** satisfy the specified sub-rule.

```yaml
matches: is-function-call
```

🔄 Matches the node if that utility rule matches it. Your gateway to modularity!

## Utility Rules Cheat Sheet

Define reusable rule definitions to cut down on duplication and build complex, maintainable rule sets.

```yaml
rules:
  - id: find-my-pattern
    rule:
      matches: my-local-check
utils:
  my-local-check:
    kind: identifier
    regex: '^my'
```

🏡 Defined within the `utils` field of your current config file. Only accessible within that file.

```yaml
# In utils/my-global-check.yml
id: my-global-check
language: javascript
rule:
  kind: variable_declarator
  has:
    kind: number_literal
```

🌍 Defined in separate YAML files in global `utilsDirs` folders, accessible across your entire project.

---

---
url: /cheatsheet/yaml.md
---
# Config Cheat Sheet

This cheat sheet provides a concise overview of ast-grep's linter rule YAML configuration.  It's designed as a handy reference for common usage.

## Basic Information

Core details that identify and define your rule and miscellaneous keys for documentation and custom data.

```yaml
id: no-console-log
```

🆔 A unique, descriptive identifier for the rule.

```yaml
language: JavaScript
```

🌐 The programming language the rule applies to.

```yaml
url: 'https://doc.link/'
```

🔗 A URL to the rule's documentation.

```yaml
metadata: { author: 'John Doe' }
```

📓 metadata	A dictionary for custom data related to the rule.

## Finding

Keys for specifying what code to search for.

```yaml
rule:
  pattern: 'console.log($$$ARGS)'
```

🎯 The core `rule` to find matching AST nodes.

```yaml
constraints:
  ARG: { kind: 'string' } }
```

⚙️ Additional `constraints` rules to filter meta-variable matches.

```yaml
utils:
  is-react:
    kind: function_declaration
    has: { kind: jsx_element }
```

🛠️ A dictionary of reusable utility rules. Use them in `matches` to modularize your rules.

## Patching

Keys for defining how to automatically fix the found code.

```yaml
transform:
  NEW_VAR:
    substring: {endChar: 1, source: $V}
```

🎩 `transform` meta-variables before they are used in `fix`.

```yaml
transform:
  NEW_VAR: substring($V, endChar=1)
```

🎩 `transform` also accepts string form.

```yaml
fix: "logger.log($$$ARGS)"
```

🔧 A `fix` string to auto-fix the matched code.

```yaml
fix:
  template: "logger.log($$$ARGS)"
  expandEnd: rule
```

🔧 Fix also accepts `FixConfig` object.

```yaml
rewriters:
- id: remove-quotes
  rule: { pattern: "'$A'" }
  fix: "$A"
```

✍️ A list of `rewriters` for complex transformations.

## Linting

Keys for configuring the messages and severity of reported issues.

```yaml
severity: warning
```

⚠️ The `severity` level of the linting message.

```yaml
message: "Avoid using $MATCH in production."
```

💬 A concise `message` explaining the rule. Matched $VAR can be used.

```yaml
note:
  Use a _logger_ instead of `console`
```

📌 More detailed `note`. It supports Markdown format.

```yaml
labels:
  ARG:
    style: 'primary'
    message: 'The argument to log'
```

🎨 Customized `labels` for highlighting parts of the matched code.

```yaml
files: ['src/**/*.js']
```

✅ Glob `files` patterns to include files for the rule.

```yaml
files:
  - glob: 'README.md'
    caseInsensitive: true
```

✅ Use object syntax for case-insensitive glob matching.

```yaml
ignores: ['test/**/*.js']
```

❌ Glob patterns to exclude files from the rule.

---

---
url: /reference/cli.md
---
# Command Line Reference

You can always see up-to-date command line options using `ast-grep --help`.
ast-grep has several subcommands as listed below.

## `ast-grep run`

Run one time search or rewrite in command line. This is the default command when you run the CLI, so `ast-grep -p 'foo()'` is equivalent to `ast-grep run -p 'foo()'`. [View detailed reference.](/reference/cli/run.html)

### Usage

```shell
ast-grep run [OPTIONS] <--pattern <PATTERN>|--kind <KIND>> [PATHS]...
```

### Arguments

`[PATHS]...`  The paths to search. You can provide multiple paths separated by spaces \[default: .]

### Options

| Short | Long | Description |
|-------|------|-------------|
| -p| --pattern `<PATTERN>` |  AST pattern to match. |
|   | --selector `<KIND>`   |  AST kind to extract sub-part of pattern to match. |
| -k| --kind `<KIND>`       |  AST kind to match. Accepts ESQuery-style selectors. See [ESQuery Style Kind](/reference/rule/esquery.html). |
| -r| --rewrite `<REWRITE>` |  String to replace the matched AST node. |
| -l| --lang `<LANG>`       |  The language of the pattern query. ast-grep will infer the language based on file extension if this option is omitted. |
|   | --debug-query`[=<format>]` |  Print query pattern's tree-sitter AST. Requires lang be set explicitly. |
|   | --strictness `<STRICTNESS>`   |  The strictness of the pattern \[possible values: cst, smart, ast, relaxed, signature, template] |
|   | --follow   |  Follow symbolic links |
|   |  --no-ignore `<NO_IGNORE>`  | Do not respect hidden file system or ignore files (.gitignore, .ignore, etc.) \[possible values: hidden, dot, exclude, global, parent, vcs] |
|   |  --stdin           | Enable search code from StdIn. See [link](/guide/tooling-overview.html#enable-stdin-mode) |
|   | --globs `<GLOBS>`   | Include or exclude file paths
| -j| --threads `<NUM>`     | Set the approximate number of threads to use \[default: heuristic]
| -i| --interactive         |  Start interactive edit session. Code rewrite only happens inside a session. |
| -U|  --update-all         |  Apply all rewrite without confirmation if true. |
|   | --json`[=<STYLE>]`    | Output matches in structured JSON  \[possible values: pretty, stream, compact] |
|   |  --color `<WHEN>`     | Controls output color \[default: auto] |
|   |  --inspect `<GRANULARITY>`  | Inspect information for file/rule discovery and scanning \[default: nothing] \[possible values: nothing, summary, entity]|
|   | --heading `<WHEN>`    | Controls whether to print the file name as heading \[default: auto] \[possible values: auto, always, never] |
| -A| --after `<NUM>`      | Show NUM lines after each match \[default: 0] |
| -B| --before `<NUM>`     | Show NUM lines before each match \[default: 0] |
| -C| --context `<NUM>`    | Show NUM lines around each match \[default: 0] |
|-h | --help                | Print help |

## `ast-grep scan`

Scan and rewrite code by configuration. [View detailed reference.](/reference/cli/scan.html)

### Usage

```shell
ast-grep scan [OPTIONS] [PATHS]...
```

### Arguments

`[PATHS]...`  The paths to search. You can provide multiple paths separated by spaces \[default: .]

### Options

| Short | Long | Description |
|-------|------|-------------|
| -c | --config `<CONFIG_FILE>`| Path to ast-grep root config, default is `sgconfig.yml` |
| -r | --rule `<RULE_FILE>`| Scan the codebase with the single rule located at the path `RULE_FILE`.|
|    | --inline-rules `<RULE_TEXT>` | Scan the codebase with a rule defined by the provided `RULE_TEXT` |
|    | --filter `<REGEX>` |Scan the codebase with rules with ids matching `REGEX` |
|    | --include-metadata | Include rule metadata in the json output |
| -j | --threads `<NUM>`   | Set the approximate number of threads to use \[default: heuristic]
| -i | --interactive|Start interactive edit session.|
| | --color `<WHEN>`|Controls output color \[default: auto] \[possible values: auto, always, ansi, never]|
| | --report-style `<REPORT_STYLE>` | \[default: rich] \[possible values: rich, medium, short]
|   | --follow   |  Follow symbolic links |
| | --json`[=<STYLE>]` | Output matches in structured JSON  \[possible values: pretty, stream, compact] |
| | --format `<FORMAT>` |  Output warning/error messages in different formats \[possible values: github, sarif] |
| -U | --update-all | Apply all rewrite without confirmation |
| | --no-ignore `<NO_IGNORE>` | Do not respect ignore files. (.gitignore, .ignore, etc.) \[possible values: hidden, dot, exclude, global, parent, vcs] |
|   |  --stdin           | Enable search code from StdIn. See [link](/guide/tooling-overview.html#enable-stdin-mode) |
|   | --globs `<GLOBS>`   | Include or exclude file paths
|   |  --inspect `<GRANULARITY>`  | Inspect information for file/rule discovery and scanning \[default: nothing] \[possible values: nothing, summary, entity]|
|   | --error`[=<RULE_ID>...]`| Set rule severity to error
|   | --warning`[=<RULE_ID>...]`| Set rule severity to warning
|   | --info`[=<RULE_ID>...]`| Set rule severity to info
|   | --hint`[=<RULE_ID>...]`| Set rule severity to hint
|   | --off`[=<RULE_ID>...]`| Turn off the rule
|   | --after `<NUM>`      | Show NUM lines after each match \[default: 0] |
|   | --before `<NUM>`     | Show NUM lines before each match \[default: 0] |
|   | --context `<NUM>`    | Show NUM lines around each match \[default: 0] |
| -A| --after `<NUM>`      | Show NUM lines after each match \[default: 0] |
| -B| --before `<NUM>`     | Show NUM lines before each match \[default: 0] |
| -C| --context `<NUM>`    | Show NUM lines around each match \[default: 0] |
| -h| --help|Print help|

## `ast-grep test`

Test ast-grep rules. See [testing guide](/guide/test-rule.html) for more details. [View detailed reference.](/reference/cli/test.html)

### Usage

```shell
ast-grep test [OPTIONS]
```

### Options

| Short | Long | Description |
|-------|------|-------------|
| -c| --config `<CONFIG>`             |Path to the root ast-grep config YAML.|
| -t| --test-dir `<TEST_DIR>`         |the directories to search test YAML files.|
|   | --snapshot-dir `<SNAPSHOT_DIR>` |Specify the directory name storing snapshots. Default to `__snapshots__`.|
|   | --skip-snapshot-tests           |Only check if the test code is valid, without checking rule output. Turn it on when you want to ignore the output of rules|
| -U| --update-all                   |Update the content of all snapshots that have changed in test.|
| -f| --filter                        |Filter rule test cases to execute using a glob pattern.|
|   | --include-off                   | Include `severity:off` rules in test
| -i| --interactive                   |start an interactive review to update snapshots selectively.|
|   | --color `<WHEN>`                |Controls output color \[default: auto] \[possible values: auto, always, ansi, never]|
| -h| --help                          |Print help.|

## `ast-grep new`

Create new ast-grep project or items like rules/tests. [View detailed reference.](/reference/cli/new.html)

### Usage

```shell
ast-grep new [COMMAND] [OPTIONS] [NAME]
```

### Commands

|Sub Command| Description|
|--|--|
| project | Create an new project by scaffolding. |
| rule    | Create a new rule. |
| test    | Create a new test case. |
| util    | Create a new global utility rule. |
| help    | Print this message or the help of the given subcommand(s). |

### Arguments

`[NAME]`  The id of the item to create.

### Options

| Short | Long | Description |
|-------|------|-------------|
| -l| `--lang <LANG>`         | The language of the item to create. |
| -y| `--yes`                 | Accept all default options without interactive input during creation. |
| -b| `--base-dir <BASE_DIR>` | Create new project/items in the folder specified by this argument `[default: .]` |
| -h| `--help`                | Print help (see more with '--help') |

## `ast-grep lsp`

Start a language server to [report diagnostics](/guide/scan-project.html) in your project. This is useful for editor integration. See [editor integration](/guide/tools/editors.html) for more details.

### Usage

```shell
ast-grep lsp
```

### Options

| Short | Long | Description |
|-------|------|-------------|
| -c | --config `<CONFIG_FILE>`| Path to ast-grep root config, default is `sgconfig.yml` |
| -h| `--help`                | Print help (see more with '--help') |

## `ast-grep completions`

Generate shell completion script.

### Usage

```shell
ast-grep completions [SHELL]
```

### Arguments

`[SHELL]`

Output the completion file for given shell.
If not provided, shell flavor will be inferred from environment.

\[possible values: bash, elvish, fish, powershell, zsh]

## `ast-grep help`

Print help message or the help of the given subcommand(s).

---

---
url: /reference/cli/run.md
---

# `ast-grep run`

Run one time search or rewrite in command line.
This is the default command when you run the CLI, so `ast-grep -p 'foo()'` is equivalent to `ast-grep run -p 'foo()'`.

## Usage

```shell
ast-grep run [OPTIONS] <--pattern <PATTERN>|--kind <KIND>> [PATHS]...
```

## Arguments

`[PATHS]...`

The paths to search. You can provide multiple paths separated by spaces

\[default: `.`]

## Run Specific Options

### `-p, --pattern <PATTERN>`

AST pattern to match

### `-r, --rewrite <FIX>`

String to replace the matched AST node

### `-l, --lang <LANG>`

The language of the pattern. For full language list, visit https://ast-grep.github.io/reference/languages.html

### `--debug-query[=<format>]`

Print query pattern's tree-sitter AST. Requires lang be set explicitly.

Possible values:

* **pattern**: Print the query parsed in Pattern format
* **ast**: Print the query in tree-sitter AST format, only named nodes are shown
* **cst**: Print the query in tree-sitter CST format, both named and unnamed nodes are shown
* **sexp**: Print the query in S-expression format

### `--selector <KIND>`

AST kind to extract sub-part of pattern to match.

selector defines the sub-syntax node kind that is the actual matcher of the pattern. See https://ast-grep.github.io/guide/rule-config/atomic-rule.html#pattern-object.

### `-k, --kind <KIND>`

AST kind to match.

`kind` accepts a single tree-sitter node kind or an ESQuery-style selector. See [ESQuery Style Kind](/reference/rule/esquery.html) for the supported selector syntax.

### `--strictness <STRICTNESS>`

The strictness of the pattern. More strict algorithm will match less code. See [match algorithm deep dive](/advanced/match-algorithm.html) for more details.

Possible values:

* **cst**:       Match exact all node
* **smart**:     Match all node except source trivial nodes
* **ast**:       Match only ast nodes
* **relaxed**:   Match ast node except comments
* **signature**: Match ast node except comments, without text
* **template**:  Similar to smart but match text only, node kinds are ignored

\[default: smart]

## Input Options

### `--no-ignore <FILE_TYPE>`

Do not respect hidden file system or ignore files (.gitignore, .ignore, etc.).

You can suppress multiple ignore files by passing `no-ignore` multiple times.

Possible values:

* **hidden**:  Search hidden files and directories. By default, hidden files and directories are skipped
* **dot**:     Don't respect .ignore files. This does *not* affect whether ast-grep will ignore files and directories whose names begin with a dot. For that, use --no-ignore hidden
* **exclude**: Don't respect ignore files that are manually configured for the repository such as git's '.git/info/exclude'
* **global**:  Don't respect ignore files that come from "global" sources such as git's `core.excludesFile` configuration option (which defaults to `$HOME/.config/git/ignore`)
* **parent**:  Don't respect ignore files (.gitignore, .ignore, etc.) in parent directories
* **vcs**:     Don't respect version control ignore files (.gitignore, etc.). This implies --no-ignore parent for VCS files. Note that .ignore files will continue to be respected

### `--stdin`

Enable search code from StdIn.

Use this if you need to take code stream from standard input.

### `--globs <GLOBS>`

Include or exclude file paths.

Include or exclude files and directories for searching that match the given glob. This always overrides any other ignore logic. Multiple glob flags may be used. Globbing rules match .gitignore globs. Precede a glob with a `!` to exclude it. If multiple globs match a file or directory, the glob given later in the command line takes precedence.

### `--follow`

Follow symbolic links.

This flag instructs ast-grep to follow symbolic links while traversing directories. This behavior is disabled by default. Note that ast-grep will check for symbolic link loops and report errors if it finds one. ast-grep will also report errors for broken links.

## Output Options

### `-i, --interactive`

Start interactive edit session.

You can confirm the code change and apply it to files selectively, or you can open text editor to tweak the matched code. Note that code rewrite only happens inside a session.

### `-j, --threads <NUM>`

Set the approximate number of threads to use.

This flag sets the approximate number of threads to use. A value of 0 (which is the default) causes ast-grep to choose the thread count using heuristics.

\[default: 0]

### `-U, --update-all`

Apply all rewrite without confirmation if true

### `--json[=<STYLE>]`

Output matches in structured JSON .

If this flag is set, ast-grep will output matches in JSON format. You can pass optional value to this flag by using `--json=<STYLE>` syntax to further control how JSON object is formatted and printed. ast-grep will `pretty`-print JSON if no value is passed. Note, the json flag must use `=` to specify its value. It conflicts with interactive.

Possible values:

* **pretty**:  Prints the matches as a pretty-printed JSON array, with indentation and line breaks. This is useful for human readability, but not for parsing by other programs. This is the default value for the `--json` option
* **stream**:  Prints each match as a separate JSON object, followed by a newline character. This is useful for streaming the output to other programs that can read one object per line
* **compact**: Prints the matches as a single-line JSON array, without any whitespace. This is useful for saving space and minimizing the output size

### `--color <WHEN>`

Controls output color.

This flag controls when to use colors. The default setting is 'auto', which means ast-grep will try to guess when to use colors. If ast-grep is printing to a terminal, then it will use colors, but if it is redirected to a file or a pipe, then it will suppress color output. ast-grep will also suppress color output in some other circumstances. For example, no color will be used if the TERM environment variable is not set or set to 'dumb'.

\[default: auto]

Possible values:

* **auto**:   Try to use colors, but don't force the issue. If the output is piped to another program, or the console isn't available on Windows, or if TERM=dumb, or if `NO_COLOR` is defined, for example, then don't use colors
* **always**: Try very hard to emit colors. This includes emitting ANSI colors on Windows if the console API is unavailable (not implemented yet)
* **ansi**:   Ansi is like Always, except it never tries to use anything other than emitting ANSI color codes
* **never**:  Never emit colors

### `--heading <WHEN>`

Controls whether to print the file name as heading.

If heading is used, the file name will be printed as heading before all matches of that file. If heading is not used, ast-grep will print the file path before each match as prefix. The default value `auto` is to use heading when printing to a terminal and to disable heading when piping to another program or redirected to files.

\[default: auto]

Possible values:

* **auto**:   Print heading for terminal tty but not for piped output
* **always**: Always print heading regardless of output type
* **never**:  Never print heading regardless of output type

### `--inspect <GRANULARITY>`

Inspect information for file/rule discovery and scanning.

This flag helps user to observe ast-grep's internal filtering of files and rules. Inspection will output how many and why files and rules are scanned and skipped. Inspection outputs to stderr and does not affect the result of the search.

The format of the output is informally defined as follows:

```
sg: <GRANULARITY>|<ENTITY_TYPE>|<ENTITY_IDENTIFIERS_SEPARATED_BY_COMMA>: KEY=VAL
```

The [Extended Backus–Naur form](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) notation is specified in the [issue](https://github.com/ast-grep/ast-grep/issues/1574).

\[default: nothing]

Possible values:

* **nothing**: Do not show any tracing information
* **summary**: Show summary about how many files are scanned and skipped
* **entity**:  Show per-file/per-rule tracing information

## Context Options

### `-A, --after <NUM>`

Show NUM lines after each match.

It conflicts with both the -C/--context flag.

\[default: 0]

### `-B, --before <NUM>`

Show NUM lines before each match.

It conflicts with both the -C/--context flag.

\[default: 0]

### `-C, --context <NUM>`

Show NUM lines around each match.

This is equivalent to providing both the -B/--before and -A/--after flags with the same value. It conflicts with both the -B/--before and -A/--after flags.

\[default: 0]

### `-h, --help`

Print help (see a summary with '-h')

## Exit codes

The program exits with status code:

* **0**: at least one line matches
* **1**: no matches found

---

---
url: /reference/cli/scan.md
---

# `ast-grep scan`

Scan and rewrite code by configuration.

## Usage

```shell
ast-grep scan [OPTIONS] [PATHS]...
```

## Arguments

`[PATHS]...`

The paths to search. You can provide multiple paths separated by spaces

\[default: .]

## Scan Specific Options

### `-c, --config <CONFIG_FILE>`

Path to ast-grep root config, default is sgconfig.yml

### `-r, --rule <RULE_FILE>`

Scan the codebase with the single rule located at the path RULE\_FILE.

This flags conflicts with --config. It is useful to run single rule without project setup.

### `--inline-rules <RULE_TEXT>`

Scan the codebase with a rule defined by the provided RULE\_TEXT.

Use this argument if you want to test a rule without creating a YAML file on disk. You can run multiple rules by separating them with `---` in the RULE\_TEXT. --inline-rules is incompatible with --rule.

### `--filter <REGEX>`

Scan the codebase with rules with ids matching REGEX.

This flags conflicts with --rule. It is useful to scan with a subset of rules from a large set of rule definitions within a project.

### `--include-metadata`

Include rule [metadata](/reference/yaml.html#metadata) in the json output.

This flags requires --json mode. Default is false.

## Input Options

### `--no-ignore <FILE_TYPE>`

Do not respect hidden file system or ignore files (.gitignore, .ignore, etc.).

You can suppress multiple ignore files by passing `no-ignore` multiple times.

Possible values:

* hidden:  Search hidden files and directories. By default, hidden files and directories are skipped
* dot:     Don't respect .ignore files. This does *not* affect whether ast-grep will ignore files and directories whose names begin with a dot. For that, use --no-ignore hidden
* exclude: Don't respect ignore files that are manually configured for the repository such as git's '.git/info/exclude'
* global:  Don't respect ignore files that come from "global" sources such as git's `core.excludesFile` configuration option (which defaults to `$HOME/.config/git/ignore`)
* parent:  Don't respect ignore files (.gitignore, .ignore, etc.) in parent directories
* vcs:     Don't respect version control ignore files (.gitignore, etc.). This implies --no-ignore parent for VCS files. Note that .ignore files will continue to be respected

### `--stdin`

Enable search code from StdIn.

Use this if you need to take code stream from standard input.

### `--follow`

Follow symbolic links.

This flag instructs ast-grep to follow symbolic links while traversing directories. This behavior is disabled by default. Note that ast-grep will check for symbolic link loops and report errors if it finds one. ast-grep will also report errors for broken links.

### `--globs <GLOBS>`

Include or exclude file paths.

Include or exclude files and directories for searching that match the given glob. This always overrides any other ignore logic. Multiple glob flags may be used. Globbing rules match .gitignore globs. Precede a glob with a `!` to exclude it. If multiple globs match a file or directory, the glob given later in the command line takes precedence.

## Output Options

### `-i, --interactive`

Start interactive edit session.

You can confirm the code change and apply it to files selectively, or you can open text editor to tweak the matched code. Note that code rewrite only happens inside a session.

### `-j, --threads <NUM>`

Set the approximate number of threads to use.

This flag sets the approximate number of threads to use. A value of 0 (which is the default) causes ast-grep to choose the thread count using heuristics.

\[default: 0]

### `-U, --update-all`

Apply all rewrite without confirmation if true

### `--json[=<STYLE>]`

Output matches in structured JSON .

If this flag is set, ast-grep will output matches in JSON format. You can pass optional value to this flag by using `--json=<STYLE>` syntax to further control how JSON object is formatted and printed. ast-grep will `pretty`-print JSON if no value is passed. Note, the json flag must use `=` to specify its value. It conflicts with interactive.

Possible values:

* pretty:  Prints the matches as a pretty-printed JSON array, with indentation and line breaks. This is useful for human readability, but not for parsing by other programs. This is the default value for the `--json` option
* stream:  Prints each match as a separate JSON object, followed by a newline character. This is useful for streaming the output to other programs that can read one object per line
* compact: Prints the matches as a single-line JSON array, without any whitespace. This is useful for saving space and minimizing the output size

### `--inspect <GRANULARITY>`

Inspect information for file/rule discovery and scanning.

This flag helps user to observe ast-grep's internal filtering of files and rules. Inspection will output how many and why files and rules are scanned and skipped. Inspection outputs to stderr and does not affect the result of the search.

The format of the output is informally defined as follows:

```
sg: <GRANULARITY>|<ENTITY_TYPE>|<ENTITY_IDENTIFIERS_SEPARATED_BY_COMMA>: KEY=VAL
```

The [Extended Backus–Naur form](https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form) notation is specified in the [issue](https://github.com/ast-grep/ast-grep/issues/1574).

\[default: nothing]

Possible values:

* **nothing**: Do not show any tracing information
* **summary**: Show summary about how many files are scanned and skipped
* **entity**:  Show per-file/per-rule tracing information

### `--format <FORMAT>`

Output warning/error messages in different formats.

Supported formats: GitHub Action, [SARIF](https://sarifweb.azurewebsites.net/) (Static Analysis Results Interchange Format).

\[possible values: github, sarif]

## Context Options

### `-A, --after <NUM>`

Show NUM lines after each match.

It conflicts with both the -C/--context flag.

\[default: 0]

### `-B, --before <NUM>`

Show NUM lines before each match.

It conflicts with both the -C/--context flag.

\[default: 0]

### `-C, --context <NUM>`

Show NUM lines around each match.

This is equivalent to providing both the -B/--before and -A/--after flags with the same value. It conflicts with both the -B/--before and -A/--after flags.

\[default: 0]

## Style Options

### `--color <WHEN>`

Controls output color.

This flag controls when to use colors. The default setting is 'auto', which means ast-grep will try to guess when to use colors. If ast-grep is printing to a terminal, then it will use colors, but if it is redirected to a file or a pipe, then it will suppress color output. ast-grep will also suppress color output in some other circumstances. For example, no color will be used if the TERM environment variable is not set or set to 'dumb'.

\[default: auto]

Possible values:

* auto:   Try to use colors, but don't force the issue. If the output is piped to another program, or the console isn't available on Windows, or if TERM=dumb, or if `NO_COLOR` is defined, for example, then don't use colors
* always: Try very hard to emit colors. This includes emitting ANSI colors on Windows if the console API is unavailable (not implemented yet)
* ansi:   Ansi is like Always, except it never tries to use anything other than emitting ANSI color codes
* never:  Never emit colors

### `--report-style <REPORT_STYLE>`

\[default: rich]

Possible values:

* rich:   Output a richly formatted diagnostic, with source code previews
* medium: Output a condensed diagnostic, with a line number, severity, message and notes (if any)
* short:  Output a short diagnostic, with a line number, severity, and message

## Rule Options

These rule option flags set the specified RULE\_ID's severity to a specific level. You can specify multiple rules by using the flag multiple times, e.g., `--error=RULE_1 --error=RULE_2`. If no RULE\_ID is provided, all rules will be set to the specified level, e.g., `--error`. Note, these flags must use `=` to specify its value.

### `--error[=<RULE_ID>...]`

Set rule severity to error

This flag sets the specified RULE\_ID's severity to error. You can specify multiple rules by using the flag multiple times. If no RULE\_ID is provided, all rules will be set to error. Note, this flag must use `=` to specify its value.

### `--warning[=<RULE_ID>...]`

Set rule severity to warning

This flag sets the specified RULE\_ID's severity to warning. You can specify multiple rules by using the flag multiple times. If no RULE\_ID is provided, all rules will be set to warning. Note, this flag must use `=` to specify its value.

### `--info[=<RULE_ID>...]`

Set rule severity to info

This flag sets the specified RULE\_ID's severity to info. You can specify multiple rules by using the flag multiple times. If no RULE\_ID is provided, all rules will be set to info. Note, this flag must use `=` to specify its value.

### `--hint[=<RULE_ID>...]`

Set rule severity to hint

This flag sets the specified RULE\_ID's severity to hint. You can specify multiple rules by using the flag multiple times. If no RULE\_ID is provided, all rules will be set to hint. Note, this flag must use `=` to specify its value.

### `--off[=<RULE_ID>...]`

Turn off rule

This flag turns off the specified RULE\_ID. You can disable multiple rules by using the flag multiple times. If no RULE\_ID is provided, all rules will be turned off. Note, this flag must use `=` to specify its value.

### `-h, --help`

Print help (see a summary with '-h')

## Built-in Rules

ast-grep provides built-in rules that can be used with the [severity override flags](#rule-options) above. These rules are not enabled by default and must be activated via CLI flags.

### `unused-suppression`

Reports `ast-grep-ignore` suppression comments that do not actually suppress any diagnostic. It behaves like a `hint` rule with auto-fix. See [Report Unused Suppressions](/guide/project/severity.html#report-unused-suppressions) for more details.

```bash
ast-grep scan --error=unused-suppression
```

### `no-suppress-all`

Reports `ast-grep-ignore` comments that do not specify a rule ID, which would suppress all diagnostics on a line. See [Disallow Suppress-All Comments](/guide/project/severity.html#disallow-suppress-all-comments) for more details.

```bash
ast-grep scan --warning=no-suppress-all
```

## Exit codes

The program exits with status code:

* **1**: if at least one rule matches (if severity setted to error)
* **0**: if no rules match

---

---
url: /reference/cli/test.md
---

# `ast-grep test`

Test ast-grep rules.

## Usage

```shell
ast-grep test [OPTIONS]
```

## Options

### `-c, --config <CONFIG>`

Path to ast-grep root config, default is sgconfig.yml

### `-t, --test-dir <TEST_DIR>`

the directories to search test YAML files

### `--snapshot-dir <SNAPSHOT_DIR>`

Specify the directory name storing snapshots. Default to **snapshots**

### `--skip-snapshot-tests`

Only check if the test code is valid, without checking rule output. Turn it on when you want to ignore the output of rules. Conflicts with --update-all

### `-U, --update-all`

Update the content of all snapshots that have changed in test. Conflicts with --skip-snapshot-tests

### `-i, --interactive`

Start an interactive review to update snapshots selectively

### `-f, --filter <FILTER>`

Filter rule test cases to execute using a glob pattern

### `--include-off`

Include `severity:off` rules in test

ast-grep will not run rules with `severity: off` by default. This option will include those rules in the test.

### `--color <WHEN>`

Controls when to use color in the output.

This is useful when piping test output to a file or another command, where ANSI escape codes would be unwanted.
Possible values:

* `auto` (default): Automatically detect if color support is available on the terminal
* `always`: Always display colors
* `ansi`: Always display colors but with ANSI escape codes (alias for `always`)
* `never`: Never display colors

### `-h, --help`

Print help

---

---
url: /reference/cli/new.md
---

# `ast-grep new`

Create new ast-grep project or items like rules/tests. Also see the step by step [guide](/guide/scan-project.html).

## Usage

```shell
ast-grep new [COMMAND] [OPTIONS] [NAME]
```

## Commands

### `project`

Create an new project by scaffolding.

By default, this command will create a root config file `sgconfig.yml`,
a rule folder `rules`, a test case folder `rule-tests` and a utility rule folder `utils`.
You can customize the folder names during the creation.

### `rule`

Create a new rule.

This command will create a new rule in one of the `rule_dirs`.
You need to provide `name` and `language` either by interactive input or via command line arguments.
ast-grep will ask you which `rule_dir` to use if multiple ones are configured in the `sgconfig.yml`.
If `-y, --yes` flag is true, ast-grep will choose the first `rule_dir` to create the new rule.

### `test`

Create a new test case.

This command will create a new test in one of the `test_dirs`.
You need to provide `name` either by interactive input or via command line arguments.
ast-grep will ask you which `test_dir` to use if multiple ones are configured in the `sgconfig.yml`.
If `-y, --yes` flag is true, ast-grep will choose the first `test_dir` to create the new test.

### `util`

Create a new global utility rule.

This command will create a new global utility rule in one of the `utils` folders.
You need to provide `name` and `language` either by interactive input or via command line arguments.
ast-grep will ask you which `util_dir` to use if multiple ones are configured in the `sgconfig.yml`.
If `-y, --yes` flag is true, ast-grep will choose the first `util_dir` to create the new item.

### `help`

Print this message or the help of the given subcommand(s)

## Arguments

`[NAME]`

The id of the item to create

## Options

### `-l, --lang <LANG>`

The language of the item to create.

This option is only available when creating rule and util.

### `-y, --yes`

Accept all default options without interactive input during creation.

You need to provide all required arguments via command line if this flag is true. Please see the command description for the what arguments are required.

### `-c, --config <CONFIG_FILE>`

Path to ast-grep root config, default is sgconfig.yml

### `-h, --help`

Print help (see a summary with '-h')

---

---
url: /reference/sgconfig.md
---

# `sgconfig.yml` Reference

## Overview

To scan a project with multiple rules, you need to specify the root of a project by maintaining a `sgconfig.yml` file.
The file is similar to `tsconfig.json` in TypeScript or `.eslintrc.js` in eslint.
You can also create the `sgconfig.yml` and related file scaffoldings by the `ast-grep new` command.

::: tip sgconfig.yml is not `rule.yml`
ast-grep has several kinds of yaml files. `sgconfig.yml` is for configuring ast-grep, like how to find rule directories or to register custom languages.
While `rule.yml` is to specify one single rule logic to find problematic code.
:::

`sgconfig.yml` has these options.

## `ruleDirs`

* type: `String`
* required: Yes

A list of string instructing where to discover ast-grep's YAML rules.

**Example:**

```yaml
ruleDirs:
- rules
- anotherRuleDir
```

Note, all items under `ruleDirs` are resolved relative to the location of `sgconfig.yml`.

## `testConfigs`

* type: `List` of `TestConfig`
* required: No

A list of object to configure ast-grep's test cases.
Each object can have two fields.

### `testDir`

* type: `String`
* required: Yes

A string specifies where to discover test cases for ast-grep.

### `snapshotDir`

* type: `String`
* required: No

A string path relative to `testDir` that specifies where to store test snapshots for ast-grep.
You can think it like `__snapshots___` in popular test framework like jest.
If this option is not specified, ast-grep will store the snapshot under the `__snapshots__` folder under the `testDir`.

Example:

```yaml
testConfigs:
- testDir: test
  snapshotDir: __snapshots__
- testDir: anotherTestDir
```

## `utilDirs`

* type: `String`
* required: No

A list of string instructing where to discover ast-grep's [global utility rules](/guide/rule-config/utility-rule.html#global-utility-rules).

## `languageGlobs`

* type: `HashMap<String, Array<String>>`
* required: No

A mapping to associate a language to files that have non-standard extensions or syntaxes.

ast-grep uses file extensions to discover and parse files in certain languages. You can use this option to support files if their extensions are not in the default mapping.

The key of this option is the language name. The values are a list of [glob patterns](https://www.wikiwand.com/en/Glob_\(programming\)) that match the files you want to process.

Note, `languageGlobs` takes precedence over the default language parser so you can reassign the parser for a specific file extension.

**Example:**

```yml
languageGlobs:
  html: ['*.vue', '*.svelte', '*.astro']
  json: ['.eslintrc']
  cpp: ['*.c'] # override the default parsers
  tsx: ['*.ts'] # useful for rule reuse
```

The above configuration tells ast-grep to treat the files with `.vue`, `.svelte`, and `.astro` extensions as HTML files, and the extension-less file `.eslintrc` as JSON files. It also overrides the default parser for C files and TS files.

:::tip Similar languages
This option can override the default language parser for a specific file extension, which is useful for rule reuse between similar languages like C/Cpp, or TS/TSX.
:::

## `customLanguages`

* type: `HashMap<String, CustomLang>`
* required: No

A dictionary of custom languages in the project.

The key of the dictionary is the custom language name. The value of the dictionary is the custom language configuration object.

Please see the [guide](/advanced/custom-language.html) for detailed instructions.

A custom language configuration object has the following options.

### `libraryPath`

* type: `String` or `HashMap<String, String>`
* required: Yes

The path to the tree-sitter dynamic library of the language. The string field is interpreted as the dynamic library path relative to the sgconfig.yml.

If `libraryPath` is a map, the key should be the [target triple](https://doc.rust-lang.org/rustc/platform-support.html) string and the value should be the dynamic library path. Projects supporting multiple host platforms can use object style configuration for different developers.

**Example:**

```yaml
# simple string is library path
libraryPath: my-lang-parser.so

# object style config
libraryPath:
  aarch64-apple-darwin: lang-parser-mac.so
  x86_64-unknown-linux-gnu: lang-parser-linux.so
# target triple list can be found below
# https://doc.rust-lang.org/rustc/platform-support.html
```

### `extensions`

* type: `Array<String>`
* required: Yes

The file extensions for this language.

### `expandoChar`

* type: `String`
* required: No

An optional char to replace $ in your pattern.

### `languageSymbol`

* type: `String`
* required: No

The dylib symbol to load ts-language, default is `tree_sitter_{name}`, e.g. `tree_sitter_mojo`.

**Example:**

```yaml
customLanguages:
  mojo:
      libraryPath: mojo.so     # path to dynamic library
      extensions: [mojo, 🔥]   # file extensions for this language
      expandoChar: _           # optional char to replace $ in your pattern
```

## `languageInjections`&#x20;

* type: `List<LanguageInjection>`
* required: No
* status: **Experimental**

A list of language injections to support embedded languages in the project like JS/CSS in HTML.
This is an experimental feature.

Please see the [guide](/advanced/language-injection.html) for detailed instructions.

A language injection object has the following options.

### `hostLanguage`

* type: `String`
* required: Yes

The host language name, e.g. `html`. This is the language of documents that contains the embedded language code.

### `rule`

* type: `Rule` object
* required: Yes

Defines the ast-grep rule to identify the injected language region within the host language documents.

### `injected`

* type: `String` or `List<String>`
* required: Yes

The injected language name, e.g. `js`. This is the language of the embedded code.

It can be a static string or a list of strings. If it is a list, ast-grep will use the `$LANG` meta variable captured in the rule to dynamically determine the injected language. The list of strings is the candidate language names to match the `$LANG` meta variable.

**Example:**

This is a configuration to support styled-components in JS files with static `injected` language.

```yaml
languageInjections:
- hostLanguage: js
  rule:
    pattern: styled.$TAG`$CONTENT`
  injected: css
```

This is a configuration to support CSS in JS style in JS files with dynamic `injected` language.

```yaml
languageInjections:
- hostLanguage: js
  rule:
    pattern: styled.$LANG`$CONTENT`
  injected: [css, scss, less]
```

---

---
url: /reference/yaml.md
---

# Configuration Reference

ast-grep's rules are written in YAML files.

One YAML file can contain multiple rules, separated by `---`.

An ast-grep rule is a YAML object with the following keys:

\[\[toc]]

## Basic Information

### `id`

* type: `String`
* required: true

Unique, descriptive identifier, e.g., `no-unused-variable`.

**Example:**

```yaml
id: no-console-log
```

### `language`

* type: `String`
* required: true

Specify the language to parse and the file extension to include in matching.

Valid values are: `Bash`, `C`, `Cpp`, `CSharp`, `Css`, `Elixir`, `Go`, `Haskell`, `Hcl`, `Html`, `Java`, `JavaScript`, `Json`, `Kotlin`, `Lua`, `Nix`, `Php`, `Python`, `Ruby`, `Rust`, `Scala`, `Solidity`, `Swift`, `Tsx`, `TypeScript`, `Yaml`

**Example:**

```yaml
language: JavaScript
```

## *Finding*

### `rule`

* type: `Rule`
* required: true

The object specify the method to find matching AST nodes. See details in [rule object reference](/reference/rule.html).

```yaml
rule:
  pattern: console.log($$$ARGS)
```

### `constraints`

* type: `HashMap<String, Rule>`
* required: false

Additional meta variables pattern to filter matches. The key is matched meta variable name without `$`. The value is a [rule object](/reference/rule.html).

**Note, constraints only applies to the single meta variable like `$ARG`,** not multiple meta variable like `$$$ARGS`.
So the key name must only refer to a single meta variable.

**Example:**

```yaml
rule:
  pattern: console.log($ARG)
constraints:
  ARG:
    kind: number
    # pattern: $A + $B
    # regex: '[a-zA-Z]+'
```

:::tip `constraints` is applied after `rule`
ast-grep will first match the `rule` while ignoring `constraints`, and then apply `constraints` to filter the matched nodes.

Constrained meta-variables usually do not work inside `not`.
:::

### `utils`

* type: `HashMap<String, Rule>`
* required: false

A dictionary of utility rules that can be used in `matches` locally.
The dictionary key is the utility rule id and the value is the rule object.
See [utility rule guide](/guide/rule-config/utility-rule).

**Example:**

```yaml
utils:
  match-function:
    any:
      - kind: function
      - kind: function_declaration
      - kind: arrow_function
```

## *Patching*

### `transform`

* type: `HashMap<String, Transformation>`
* required: false

A dictionary to manipulate meta-variables. The dictionary key is the new variable name.
The dictionary value is a transformation object or transformation string that specifies how meta var is processed.

Please also see [transformation reference](/reference/yaml/transformation) for details.

**Example:**

```yaml
transform:
  NEW_VAR_NAME:      # new variable name
    replace:         # transform operation
      source: $ARGS
      replace: '^.+'
      by: ', '

# string style for ast-grep 0.38.3+
transform:
  NEW_VAR_NAME: replace($ARGS, replace='^.+', by=', ')
```

### `fix`

* type: `String` or `FixConfig`
* required: false

A pattern or a `FixConfig` object to auto fix the issue. See details in [fix object reference](/reference/yaml/fix.html).

It can reference meta variables that appeared in the rule.

**Example:**

```yaml
fix: logger.log($$$ARGS)

# you can also use empty string to delete match
fix: ""
```

### `rewriters`

* type: `Array<Rewriter>`
* required: false

A list of rewriter rules that can be used in [`rewrite` transformation](/reference/yaml/transformation.html#rewrite).

A rewriter rule is similar to ordinary YAML rule, but it ony contains *finding* fields, *patching* fields and `id`.

Please also see [rewriter reference](/reference/yaml/rewriter.html) for details.

**Example:**

```yaml
rewriters:
- id: stringify
  rule: { pattern: "'' + $A" }
  fix: "String($A)"
  # you can also use these fields
  # transform, utils, constraints
```

## Linting

### `severity`

* type: `String`
* required: false

Specify the level of matched result. Available choice: `hint`, `info`, `warning`, `error` or `off`.

When `severity` is `off`, ast-grep will disable the rule in scanning.

**Example:**

```yaml
severity: warning
```

### `message`

* type: `String`
* required: false

Main message highlighting why this rule fired. It should be single line and concise,
but specific enough to be understood without additional context.

It can reference meta-variables that appeared in the rule.

**Example:**

```yaml
message: "console.log should not be used in production code"
```

### `note`

* type: `String`
* required: false

Additional notes to elaborate the message and provide potential fix to the issue.

`note` can contains markdown syntax, but it *cannot* reference meta-variables.

**Example:**

```yaml
note: "Use a logger instead"
```

### `labels`

* type: `HashMap<String, LabelConfig>`
* required: false

A dictionary of labels to customize highlighting. The dictionary key is the meta-variable name without `$`, defined in `rules` or `constraints`. The value is a label config object containing the following fields:

* `style`: (required) the style of the label. Available choice: `primary`, `secondary`.
* `message`: (optional) the message to be displayed in the editor extension.

**Example:**

```yaml
labels:
  ARG:
    style: primary
    message: "This is the argument"
  FUNC:
    style: secondary
    message: "This is the function"
```

Please also see [label guide](/guide/project/lint-rule.html#customize-code-highlighting) for details.

## Globbing

### `files`

* type: `Array<Glob>`
* required: false

Glob patterns to specify that the rule only applies to matching files. It is tested if `ignores` does not exist or a file does not match any `ignores` glob.

Each item in the array can be either:

* A **string**: a simple glob pattern
* An **object** with the following fields:
  * `glob` (required): the glob pattern string
  * `caseInsensitive` (optional): whether the glob matching is case insensitive. Defaults to `false`.

**Example:**

```yaml
# Simple string globs
files:
  - src/**/*.js
  - src/**/*.ts
```

```yaml
# Object syntax with case-insensitive matching
files:
  - glob: '*.ts'
    caseInsensitive: true
  - glob: 'README.md'
    caseInsensitive: true
```

```yaml
# Mixed formats
files:
  - '*.ts'                          # simple string
  - glob: 'README.md'               # object with case-insensitive
    caseInsensitive: true
  - 'src/**/*.tsx'                  # simple string
```

:::warning Don't add `./`
Be sure to remove `./` to the beginning of your rules. ast-grep will not recognize the paths if you add `./`.
:::

Paths in `files` are relative to the project root directory, that is, `sgconfig.yml`'s directory.

### `ignores`

* type: `Array<Glob>`
* required: false

Glob patterns that exclude rules from applying to files. A file is tested against `ignores` list before matching `files`.

Like `files`, each item can be either a string or an object with `glob` and `caseInsensitive` fields.

**Example:**

```yaml
# Simple string globs
ignores:
  - test/**/*.js
  - test/**/*.ts
```

```yaml
# With case-insensitive matching
ignores:
  - 'test/**'
  - glob: 'BUILD'
    caseInsensitive: true
```

A typical globing process works as follows:

1. If `ignores` is configured, a file will be skipped if it matches any of the glob in the list(`files` will not be tested).
2. If `files` is configured, a file will be included if and only if it matches one of the glob in the list.
3. If neither `ignores`/`files` is configured, a file is included by default.

:::warning `ignores` in YAML is different from `--no-ignore` in CLI
ast-grep respects common ignore files like `.gitignore` and hidden files by default.
To disable this behavior, use [`--no-ignore`](/reference/cli.html#scan) in CLI.
`ignores` is a rule-wise configuration that only filters files that are not ignored by the CLI.
:::

Paths in `ignores` are relative to the project root directory, that is, `sgconfig.yml`'s directory.

## Other

### `url`

* type: `String`
* required: false

Documentation link to this rule. It will be displayed in editor extension if supported.

**Example:**

```yaml
url: 'https://ast-grep.github.io/catalog/python/#migrate-openai-sdk'
```

### `metadata`

* type: `HashMap<String, String>`
* required: false

Extra information for the rule. This section can include custom data for external program to consume. For example, CVE/OWASP information can be added here for security research.

ast-grep will output `metadata` with matches in [`--json`](/reference/cli/scan.html#json-style) mode if [`--include-metadata`](/reference/cli/scan.html#include-metadata) is on.

**Example:**

```yaml
metadata:
  extraField: 'Extra information for other usages'
  complexData:
    key: value
```

---

---
url: /reference/yaml/fix.md
---
# Fix

ast-grep has two kinds of fixes: `string` and `FixConfig`.

## String Fix

* type: `String`

A string fix is a string that will be used to replace the matched AST node.

You can use meta variables in the string fix to reference the matched AST node.

N.B. Fix string is not parsed by tree-sitter. So meta variables can appear anywhere in the string.

Example:

```yaml
rule:
  pattern: console.log($$$ARGS)
fix: logger.log($$$ARGS)
```

## `FixConfig`

* type: `Object`

A `FixConfig` is an advanced object configuration that specifies how to fix the matched AST node.

ast-grep rule can only fix one target node at one time by replacing the target node text with a new string.
This works fine for function statement/calls but it has always been problematic for list-item like items in an array, key-value pairs in a dictionary. We cannot delete an item completely because we also need to delete the surrounding comma.

`FixConfig` is designed to solve this problem. It allows you to specify a template string and two additional rules to expand the matched AST node to the start and end of the matched AST node.

It has the following fields:

### `template`

* type: `String`

This is the same as the string fix.

### `expandStart`

* type: `Rule`

A rule object, which is a rule object with one additional field `stopBy`.

The fixing range's start will be expanded until the rule is not met.

### `expandEnd`

* type: `Rule`

A rule object, which is a rule object with one additional field `stopBy`.

The fixing range' end start will be expanded until the rule is not met.

Example:

```yaml
rule:
  kind: pair
  has:
    field: key
    regex: Remove
# remove the key-value pair and its comma
fix:
  template: ''
  expandEnd: { regex: ',' } # expand the range to the comma
```

---

---
url: /reference/yaml/transformation.md
---

# Transformation Object

A transformation object is used to manipulate meta variables. It is a dictionary with the following structure:

* a **key** that specifies which string operation will be applied to the meta variable, and
* a **value** that is another object with the details of how to perform the operation.

Different string operation keys expect different object values.

## `replace`

Use a regular expression to replace the text in a meta-variable with a new text.

`replace` transformation expects an object value with the following properties:

### `replace`

* type: `String`
* required: true

A Rust regular expression to match the text to be replaced.

### `by`

* type: `String`
* required: true

A string to replace the matched text.

### `source`

* type: `String`
* required: true

A meta-variable name to be replaced.

*The meta-variable name must be prefixed with `$`.*

**Example**:

```yaml
transform:
  NEW_VAR:
    replace:
      replace: regex
      by: replacement
      source: $VAR

# string style for ast-grep 0.38.3+
transform:
  NEW_VAR: replace($VAR, replace=regex, by=replacement)
```

:::tip Pro tip
You can use regular expression capture groups in the `replace` field and refer to them in the `by` field. See [replace guide](/guide/rewrite/transform.html#rewrite-with-regex-capture-groups)
:::

## `substring`

Create a new string by cutting off leading and trailing characters.

`substring` transformation expects an object value with the following properties:

### `startChar`

* type: `Integer`
* required: false

The starting character index of the new string, **inclusively**.
If omitted, the new string starts from the beginning of the source string.
The index can be negative, in which case the index is counted from the end of the string.

### `endChar`

* type: `Integer`
* required: false

The ending character index of the new string, **exclusively**.
If omitted, the new string ends at the end of the source string.
The index can be negative, in which case the index is counted from the end of the string.

### `source`

* type: `String`
* required: true

A meta-variable name to be truncated.

*The meta-variable name must be prefixed with `$`.*

**Example**:

```yaml
transform:
  NEW_VAR:
    substring:
      startChar: 1
      endChar: -1
      source: $VAR

# string style for ast-grep 0.38.3+
transform:
  NEW_VAR: substring($VAR, startChar=1, endChar=-1)
```

:::tip Pro Tip
`substring` works like [Python's string slicing](https://www.digitalocean.com/community/tutorials/python-slice-string).
They both have inclusive start, exclusive end and support negative index.

`substring`'s index is based on unicode character count, instead of byte.
:::

## `convert`

Change the string case of a meta-variable, such as from `camelCase` to `underscore_case`.

This transformation is inspired by TypeScript's [intrinsic string manipulation type](https://www.typescriptlang.org/docs/handbook/2/template-literal-types.html#intrinsic-string-manipulation-types).

Ideally, the source string should be an identifier in the rule language.

`convert` transformation expects an object value with the following properties:

### `toCase`

* type: `StringCase`
* required: true

The target case to convert to.

Some string cases will first split the source string into words, then convert each word's case, and finally join the words back together. You can fine-tune the behavior of these separator-sensitive string cases by the `separatedBy` option.

ast-grep supports the following cases:

#### `StringCase`

|Name|Example input|Example output|Separator sensitive?|
|---|---:|---:|--:|
|`lowerCase`| astGrep| astgrep| No|
|`upperCase`| astGrep| ASTGREP| No|
|`capitalize`| astGrep| AstGrep| No|
|`camelCase`| ast\_grep| astGrep| Yes|
|`snakeCase`| astGrep| ast\_grep| Yes|
|`kebabCase`| astGrep| ast-grep| Yes|
|`pascalCase`| astGrep| AstGrep| Yes|

### `separatedBy`

* type: `Array<Separator>`
* required: false
* default: all separators

A list of separators to be used to separate words in the source string.

ast-grep supports the following separators:

#### `Separator`

|Name|Separator character |Example input|Example output|
|---|:---:|:---:|:---:|
|`Dash`|`-`| ast-grep| \[ast, grep]|
|`Dot`|`.`| ast.grep| \[ast, grep]|
|`Space`|` `| ast grep| \[ast, grep]|
|`Slash`|`/`| ast/grep| \[ast, grep]|
|`Underscore`|`_`| ast\_grep| \[ast, grep]|
|`CaseChange`|Described below| astGrep| \[ast, grep]|

`CaseChange` separator is a special separator that splits the string when two consecutive characters' case changed.
More specifically, it splits the string in the following two scenarios.

* At the position between a lowercase letter and an uppercase letter, e.g. `astGrep` -> `[ast, Grep]`
* Before an uppercase letter that is not the first character and is followed by a lowercase letter, e.g. `ASTGrep` -> `[AST, Grep]`

More examples are shown below. You can also inspect [the equivalent regular expression examples](https://regexr.com/7prq5) to see how `CaseChange` works in action

```
RegExp -> [Reg, Exp]
XMLHttpRequest -> [XML, Http, Request]
regExp -> [reg, Exp]
writeHTML -> [write, HTML]
```

### `source`

* type: `String`
* required: true

A meta-variable name to convert.

*The meta-variable name must be prefixed with `$`.*

**Example**:

```yaml
transform:
  NEW_VAR:
    convert:
      toCase: kebabCase
      separatedBy: [underscore]
      source: $VAR

# string style for ast-grep 0.38.3+
transform:
  NEW_VAR: convert($VAR, toCase=kebabCase, separatedBy=[underscore])
```

Suppose we have a string `ast_Grep` as the input `$VAR`, The example above will convert the string as following:

* split the string by `_` into `[ast, Grep]`
* convert the words to lowercase words `[ast, grep]`
* join the words by `-` into the target string `ast-grep`

Thank [Aarni Koskela](https://github.com/akx) for proposing and implementing the first version of this feature!

## `rewrite`

`rewrite` is an experimental transformation that allows you to selectively transform a meta variable by `rewriter` rules.
Instead of rewriting the single target node which matches the rule, `rewrite` can rewrite a subset of AST captured by a meta-variable.

Currently, it is an experimental feature. Please see the [issue](https://github.com/ast-grep/ast-grep/issues/723)

`rewrite` transformation expects an object value with the following properties:

### `source`

* type: `String`
* required: true

The meta-variable name to be rewritten.

*The meta-variable can be single meta-variable, prefixed with `$`, or multiple prefixed with `$$$$`.*

ast-grep will find matched descendants nodes of the source meta-variable for single meta-variable and apply the rewriter rules to them.
For multiple meta-variables, ast-grep will find matched descendants nodes of each node in the meta-variable list.

### `rewriters`

* type: `Array<String>`
* required: true

A list of rewriter rules to apply to the source meta-variable. The rewrite rules work like ast-grep's fix mode.

`rewriters` can only refer to the rules specified in [`rewriters`](/reference/yaml/rewriter.html) [section](/reference/yaml.html#rewriters).

ast-grep will find nodes in the meta-variable's AST that match the rewriter rules, and rewrite them to the `fix` string/object in the matched rule.

`rewriter` rules will not have overlapping matches. Nodes on the higher level of AST, or closer to the root node, will be matched first.

For one single node, `rewriters` are matched in order, and only the first match will be applied. Subsequent rules will be ignored.

### `joinBy`

* type: `String`
* required: false

By default, the rewritten nodes will be put back to the original syntax tree.

If you want to aggregate the rewrite in other fashion, you can specify a string to join the rewritten nodes. For example, you can join generated statements using new line.

**Example**:

```yaml
transform:
  NEW_VAR:
    rewrite:
      source: $VAR
      rewriters: [rule1, rule2]
      joinBy: "\n"

# string style for ast-grep 0.38.3+
transform:
  NEW_VAR: rewrite($VAR, rewriters=[rule1, rule2], joinBy='\n')
```

Thank [Eitan Mosenkis](https://github.com/emosenkis) for proposing this idea!

---

---
url: /reference/yaml/rewriter.md
---
# Rewriter

Rewriter is a powerful, and experimental, feature that allows you to manipulate the code in a more complex way.

ast-grep rule has a `rewriters` field which is a list of rewriter objects that can be used to transform code of specific nodes matched by meta-variables.

A rewriter rule is similar to ordinary ast-grep rule, except that:

* It only has `id`, `rule`, `constraints`, `transform`, `utils`, and `fix` fields.
* `id`, `rule` and `fix` are required in rewriter.
* `rewriters` can only be used in [`rewrite`](/reference/yaml/transformation.html#rewrite) transformation.
* Meta-variables defined in one `rewriter` are not accessible in other `rewriter` or the original rule.
* `utils` and `transform` are independent similar to meta-variables. That is, these two fields can only be used by the defining rewriter.
* You can use other rewriters in a rewriter rule's `transform` section if they are defined in the same `rewriter` list.

:::warning Consider ast-grep API
Rewriters are an advanced feature and should be used with caution, and it is experimental at the moment. If possible, you can use ast-grep's [API](/guide/api-usage.html) as an alternative.
:::

Please ask questions on [StackOverflow](https://stackoverflow.com/questions/tagged/ast-grep), [GitHub Discussions](https://github.com/ast-grep/ast-grep/discussions) or [discord](https://discord.com/invite/4YZjf6htSQ) for help.

## `id`

* type: `String`
* required: true

## `rule`

* type: `Rule`
* required: true

The object specify the method to find matching AST nodes. See details in [rule object reference](/reference/rule.html).

## `fix`

* type: `String` or `FixConfig`
* required: true

A pattern or a `FixConfig` object to auto fix the issue. See details in [fix object reference](/reference/yaml/fix.html).

## `constraints`

* type: `HashMap<String, Rule>`
* required: false

Additional meta variables pattern to filter matches. The key is matched meta variable name without `$`. The value is a [rule object](/reference/rule.html).

## `transform`

* type: `HashMap<String, Transformation>`
* required: false

A dictionary to manipulate meta-variables. The dictionary key is the new variable name.
The dictionary value is a transformation object that specifies how meta var is processed.

**Note: variables defined `transform` are only available in the `rewriter` itself.**

## `utils`

* type: `HashMap<String, Rule>`
* required: false

A dictionary of utility rules that can be used in `matches` locally.
The dictionary key is the utility rule id and the value is the rule object.
See [utility rule guide](/guide/rule-config/utility-rule).

**Note: util rules defined `transform` are only available in the `rewriter` itself.**

## Example

Suppose we want to rewrite a [barrel](https://vercel.com/blog/how-we-optimized-package-imports-in-next-js) [import](https://marvinh.dev/blog/speeding-up-javascript-ecosystem-part-7/) to individual imports in JavaScript. For example,

```JavaScript
import { A, B, C } from './module';
// rewrite the above to
import A from './module/a';
import B from './module/b';
import C from './module/c';
```

It is impossible to do this in ast-grep YAML without rewriters because ast-grep can only replace one node at a time with a string. We cannot process multiple imported identifiers like `A, B, C`.

However, rewriter rules can be applied to captured meta-variables' descendant nodes, which can achieve the *multiple node processing*.

**Our first step is to write a rule to capture the import statement.**

```yaml
rule:
  pattern: import {$$$IDENTS} from './module'
```

This will capture the imported identifiers `A, B, C` in `$$$IDENTS`.

**Next, we need to transform `$$$IDENTS` to individual imports.**

The idea is that we can find the identifier nodes in the `$$$IDENT` and rewrite them to individual imports.

```yaml
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  fix: import $IDENT from './module/$IDENT'
```

The `rewrite-identifier` above will rewrite the identifier node to individual imports. To illustrate, the rewriter will change identifier `A` to  `import A from './module/A'`.

Note the library path has the uppercase letter `A` same as the identifier at the end, but we want it to be a lowercase letter in the import statement.
The [`convert`](/reference/yaml/transformation.html#convert) operation in `transform` can be helpful in the rewriter rule as well.

```yaml
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  transform:
    LIB: { convert: { source: $IDENT, toCase: lowerCase } }
  fix: import $IDENT from './module/$LIB'
```

**We can now apply the rewriter to the matched variable `$$$IDENTS`.**

The `rewrite` will find identifiers in `$$$IDENTS`, as specified in `rewrite-identifier`'s rule,
and rewrite it to single import statement.

```yaml
transform:
  IMPORTS:
    rewrite:
      rewriters: [rewrite-identifer]
      source: $$$IDENTS
      joinBy: "\n"
```

Note the `joinBy` field in the `transform` section. It is used to join the rewritten import statements with a newline character.

**Finally, we can use the `IMPORTS` in the `fix` field to replace the original import statement.**

The final rule will be like this.

```yaml
id: barrel-to-single
language: JavaScript
rule:
  pattern: import {$$$IDENTS} from './module'
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  transform:
    LIB: { convert: { source: $IDENT, toCase: lowerCase } }
  fix: import $IDENT from './module/$LIB'
transform:
  IMPORTS:
    rewrite:
      rewriters: [rewrite-identifer]
      source: $$$IDENTS
      joinBy: "\n"
fix: $IMPORTS
```

See the [playground link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBpbXBvcnQgeyQkJElERU5UU30gZnJvbSAnLi9tb2R1bGUnXG5yZXdyaXRlcnM6XG4tIGlkOiByZXdyaXRlLWlkZW50aWZlclxuICBydWxlOlxuICAgIHBhdHRlcm46ICRJREVOVFxuICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgdHJhbnNmb3JtOlxuICAgIExJQjogeyBjb252ZXJ0OiB7IHNvdXJjZTogJElERU5ULCB0b0Nhc2U6IGxvd2VyQ2FzZSB9IH1cbiAgZml4OiBpbXBvcnQgJElERU5UIGZyb20gJy4vbW9kdWxlLyRMSUInXG50cmFuc2Zvcm06XG4gIElNUE9SVFM6XG4gICAgcmV3cml0ZTpcbiAgICAgIHJld3JpdGVyczogW3Jld3JpdGUtaWRlbnRpZmVyXVxuICAgICAgc291cmNlOiAkJCRJREVOVFNcbiAgICAgIGpvaW5CeTogXCJcXG5cIlxuZml4OiAkSU1QT1JUUyIsInNvdXJjZSI6ImltcG9ydCB7IEEsIEIsIEMgfSBmcm9tICcuL21vZHVsZSc7In0=) for the complete example.

---

---
url: /reference/rule.md
---

# Rule Object Reference

A rule object can have these keys grouped in three categories:

\[\[toc]]

Atomic rules are the most basic rules to match AST nodes. Relational rules filter matched target according to their position relative to other nodes. Composite rules use logic operation all/any/not to compose the above rules to larger rules.

All of these keys are optional. However, at least one of them must be present and **positive**.

A rule is called **positive** if it only matches nodes with specific kinds. For example, a `kind` rule is positive because it only matches nodes with the kind specified by itself. A `pattern` rule is positive because the pattern itself has a kind and the matching node must have the same kind. A `regex` rule is not positive though because it matches any node as long as its text satisfies the regex.

## Atomic Rules

### `pattern`

* type: `String` or `Object`

A `String` pattern will match one single AST node according to [pattern syntax](/guide/pattern-syntax).

**Example:**

```yml
pattern: console.log($ARG)
```

`pattern` also accepts an `Object` with `context`,`selector` and optionally `strictness`.

By default `pattern` parses code as a standalone file. You can use the `selector` field  to pull out the specific part to match.

**Example**:

We can select class field in JavaScript by this pattern.

```yml
pattern:
  selector: field_definition
  context: class { $F }
```

***

You can also use `strictness` to change the matching algorithm of pattern. See the [deep dive doc](/advanced/match-algorithm.html) for more detailed explanation for strictness.

**Example**:

```yml
pattern:
  context: foo($BAR)
  strictness: relaxed
```

`strictness` accepts these options: `cst`, `smart`, `ast`, `relaxed` and `signature`.

### `kind`

* type: `String`

The kind name of the node to match. You can look up code's kind names in [playground](/playground).

**Example:**

```yml
kind: call_expression
```

ast-grep 0.39+ also supports limited ESQuery syntax for `kind`:

**Example:**

```yml
kind: call_expression > identifier
```

See [ESQuery style kind](/reference/rule/esquery.html) for supported selectors and their equivalent rule objects.

### `regex`

* type: `String`

A [Rust regular expression](https://docs.rs/regex/latest/regex/) to match the node's text. The regex must match the whole text of the node.

> Its syntax is similar to Perl-style regular expressions, but lacks a few features like look around and backreferences.

Example:

::: code-group

```yml [Literal]
regex: console
```

```yml [Character Class]
regex: ^[a-z]+$
```

```yml [Flag]
regex: (?i)a(?-i)b+
```

:::

### `nthChild`

* type: `number | string | Object`

`nthChild` finds nodes based on their indexes in the parent node's children list.

It can accept either a number, a string or an object:

* number: match the exact nth child
* string: `An+B` style string to match position based on formula
* object: nthChild object has several options to tweak the behavior of the rule
  * `position`: a number or an An+B style string
  * `reverse`: boolean indicating if count index from the end of sibling list
  * `ofRule`: object to filter the sibling node list based on rule

**Example:**

```yaml
# a number to match the exact nth child
nthChild: 3

# An+B style string to match position based on formula
nthChild: 2n+1

# object style nthChild rule
nthChild:
  # accepts number or An+B style string
  position: 2n+1
  # optional, count index from the end of sibling list
  reverse: true # default is false
  # optional, filter the sibling node list based on rule
  ofRule:
    kind: function_declaration # accepts ast-grep rule
```

**Note:**

* nthChild is inspired the [nth-child CSS selector](https://developer.mozilla.org/en-US/docs/Web/CSS/:nth-child).
* nthChild's index is 1-based, not 0-based, as in the CSS selector.
* nthChild's node list only includes named nodes, not unnamed nodes.

### `range`

* type: `RangeObject`

A `RangeObject` is an object with two fields `start` and `end`, each of which is an object with two fields `line` and `column`.

Both `line` and `column` are 0-based and character-based. `start` is inclusive and `end` is exclusive.

**Example:**

```yml
range:
  start:
    line: 0
    column: 0
  end:
    line: 0
    column: 3
```

The above example will match an AST node having the first three characters of the first line like `foo` in `foo.bar()`.

## Relational Rules

### `inside`

* type: `Object`

A relational rule object, which is a `Rule` object with two additional fields `stopBy` and `field`.

The target node must appear inside of another node matching the `inside` sub-rule.

Example:

```yaml
inside:
  pattern: class $TEST { $$$ } # a sub rule object
  stopBy: end                  # stopBy accepts 'end', 'neighbor' or another rule object.
  field: body                  # specify the sub-node in the target
```

Please refer to [relational rule guide](/guide/rule-config/relational-rule) for detailed explanation of `stopBy` and `field`.

### `has`

* type: `Object`

A relational rule object, which is a `Rule` object with two additional fields `stopBy` and `field`.

The target node must has a descendant node matching the `has` sub-rule.

Example:

```yaml
has:
  kind: property_identifier    # a sub rule object
  stopBy: end                  # stopBy accepts 'end', 'neighbor' or another rule object.
  field: name                  # specify the sub-node in the target
```

Please refer to [relational rule guide](/guide/rule-config/relational-rule) for detailed explanation of `stopBy` and `field`.

### `precedes`

* type: `Object`

A relational rule object, which is a `Rule` object with one additional field `stopBy`.

The target node must appear before another node matching the `precedes` sub-rule.

Note `precedes` does not have `field` option.

Example:

```yml
precedes:
  kind: function_declaration   # a sub rule object
  stopBy: end                  # stopBy accepts 'end', 'neighbor' or another rule object.
```

### `follows`

* type: `Object`

A relational rule object, which is a `Rule` object with one additional field `stopBy`.

The target node must appear after another node matching the `follows` sub-rule.

Note `follows` does not have `field` option.

Example:

```yml
follows:
  kind: function_declaration   # a sub rule object
  stopBy: end                  # stopBy accepts 'end', 'neighbor' or another rule object.
```

***

There are two additional fields in relational rules:

#### `stopBy`

* type: `"neighbor"` or `"end"` or `Rule` object
* default: `"neighbor"`

`stopBy` is an option to control how the search should stop when looking for the target node.

It can have three types of value:

* `"neighbor"`: stop when the target node's immediate surrounding node does not match the relational rule. This is the default behavior.
* `"end"`: search all the way to the end of the search direction. i.e. to the root node for `inside`, to the leaf node for `has`, to the first sibling for `follows`, and to the last sibling for `precedes`.
* `Rule` object: stop when the target node's surrounding node does match the rule. `stopBy` is inclusive. If the matching surrounding node also match the relational rule, the target node is still considered as matched.

#### `field`

* type: `String`
* required: No
* Only available in `inside` and `has` relational rules

`field` is an option to specify the sub-node in the target node to match the relational rule.

Note `field` and `kind` are two different concepts.

:::tip
Only relational rules have `stopBy` and `field` options.
:::

## Composite Rules

### `all`

* type: `Array<Rule>`

`all` takes a list of sub rules and matches a node if all of sub rules match.
The meta variables of the matched node contain all variables from the sub rules.

Example:

```yml
all:
  - kind: call_expression
  - pattern: console.log($ARG)
```

### `any`

* type: `Array<Rule>`

`any` takes a list of sub rules and matches a node if any of sub rules match.
The meta variables of the matched node only contain those of the matched sub rule.

Example:

```yml
any:
  - pattern: console.log($ARG)
  - pattern: console.warn($ARG)
  - pattern: console.error($ARG)
```

:::warning all/any refers to rules, not nodes
`all` will match a node only if all sub rules must match.

It will never match multiple nodes at once. Use it with other rules like `has`/`inside` will not alter this behavior.
See the [composite rule guide](/guide/rule-config/composite-rule.html#all-and-any-refers-to-rules-not-nodes) for more details and examples.
:::

### `not`

* type: `Object`

`not` takes a single sub rule and matches a node if the sub rule does not match.

Example:

```yml
not:
  pattern: console.log($ARG)
```

### `matches`

* type: `String`

`matches` takes a utility rule id and matches a node if the utility rule matches. See [utility rule guide](/guide/rule-config/utility-rule) for more details.

Example:

```yml
utils:
  isFunction:
    any:
      - kind: function_declaration
      - kind: function
rule:
  matches: isFunction
```

---

---
url: /reference/rule/esquery.md
---

# ESQuery Style Kind

ast-grep supports limited ESQuery style selector in the `kind` rule.

The selector is still written in the `kind` field. ast-grep will parse the selector and convert it to the corresponding rule object internally.

```yaml
rule:
  kind: call_expression > identifier
```

The example above will match the `identifier` node that is a direct child of `call_expression`.

## Basic Selector

A single node kind is the same as a normal `kind` rule.

```yaml
kind: identifier
```

is equivalent to

```yaml
kind: identifier
```

## Relationship Selectors

### Child Selector

The `>` selector matches a direct child node.

```yaml
kind: call_expression > identifier
```

is equivalent to

```yaml
kind: identifier
inside:
  kind: call_expression
```

### Descendant Selector

A space selector matches a descendant node.

```yaml
kind: call_expression identifier
```

is equivalent to

```yaml
kind: identifier
inside:
  kind: call_expression
  stopBy: end
```

### Adjacent Sibling Selector

The `+` selector matches the next sibling node.

```yaml
kind: decorator + method_definition
```

is equivalent to

```yaml
kind: method_definition
follows:
  kind: decorator
```

### Following Sibling Selector

The `~` selector matches a following sibling node.

```yaml
kind: decorator ~ method_definition
```

is equivalent to

```yaml
kind: method_definition
follows:
  kind: decorator
  stopBy: end
```

## Comma Selector

Comma separated selectors are converted to `any`.

```yaml
kind: identifier, number
```

is equivalent to

```yaml
any:
  - kind: identifier
  - kind: number
```

## Pseudo-classes

### `:has`

`:has` matches a node if the node has a descendant matching the inner selector.

```yaml
kind: function_declaration:has(return_statement)
```

is equivalent to

```yaml
kind: function_declaration
has:
  kind: return_statement
  stopBy: end
```

You can use `>` in `:has` to match a direct child.

```yaml
kind: expression_statement:has(> call_expression)
```

is equivalent to

```yaml
kind: expression_statement
has:
  kind: call_expression
```

### `:not`

`:not` negates the inner selector.

```yaml
kind: identifier:not(number)
```

is equivalent to

```yaml
kind: identifier
not:
  kind: number
```

### `:is`

`:is` accepts comma separated selectors and is converted to `any`.

```yaml
kind: :is(identifier, number)
```

is equivalent to

```yaml
any:
  - kind: identifier
  - kind: number
```

It can be combined with relationship selectors.

```yaml
kind: call_expression > :is(identifier, number)
```

is equivalent to

```yaml
all:
  - any:
      - kind: identifier
      - kind: number
  - inside:
      kind: call_expression
```

### `:nth-child`

`:nth-child` maps to ast-grep's [`nthChild`](/reference/rule.html#nthchild) rule.

```yaml
kind: array > number:nth-child(2n+1)
```

is equivalent to

```yaml
all:
  - kind: number
  - nthChild: 2n+1
  - inside:
      kind: array
```

`:nth-child` also supports `of` syntax.

```yaml
kind: array > :nth-child(1 of number)
```

is equivalent to

```yaml
all:
  - nthChild:
      position: 1
      ofRule:
        kind: number
  - inside:
      kind: array
```

```yaml
kind: array > :nth-child(2n+1 of number)
```

is equivalent to

```yaml
all:
  - nthChild:
      position: 2n+1
      ofRule:
        kind: number
  - inside:
      kind: array
```

### `:nth-last-child`

`:nth-last-child` is equivalent to `nthChild` with `reverse: true`.

```yaml
kind: array > number:nth-last-child(1)
```

is equivalent to

```yaml
all:
  - kind: number
  - nthChild:
      position: 1
      reverse: true
  - inside:
      kind: array
```

## Compound Selector

Compound selectors are combined with `all`.

```yaml
kind: function_declaration:has(return_statement):not(generator_function)
```

is equivalent to

```yaml
all:
  - kind: function_declaration
  - has:
      kind: return_statement
      stopBy: end
  - not:
      kind: generator_function
```

## Current Limitations

The ESQuery style `kind` only supports part of ESQuery syntax:

* Class selectors like `.body` are tokenized but rejected as unsupported.
* Supported pseudo-classes are only `has`, `not`, `is`, `nth-child` and `nth-last-child`.
* `:has(...)`, `:not(...)` and `of ...` parse a single complex selector, not a comma selector list.
* `:is(...)` is the one pseudo-class that accepts comma separated selector lists.
* Identifiers can include letters, digits, `_` and `-`, but cannot start with a digit.

---

---
url: /reference/api.md
---
# API Reference

ast-grep currently has an experimental API for [Node.js](https://nodejs.org/).

You can see [API usage guide](/guide/api-usage.html) for more details.

\[\[toc]]

## NAPI

Please see the link for up-to-date type declaration.

https://github.com/ast-grep/ast-grep/blob/main/crates/napi/index.d.ts

### Supported Languages

`@ast-grep/napi` supports JS ecosystem languages by default.
More custom languages can be loaded via [`registerDynamicLanguage`](https://github.com/search?q=repo%3Aast-grep%2Flangs%20registerDynamicLanguage\&type=code).

#### Type

```ts
export const enum Lang {
  Html = 'Html',
  JavaScript = 'JavaScript',
  Tsx = 'Tsx',
  Css = 'Css',
  TypeScript = 'TypeScript',
}

// More custom languages can be loaded
// see https://github.com/ast-grep/langs
type CustomLang = string & {}
```

`CustomLang` is not widely used now. If you have use case and needs support, please file an issue in the [@ast-grep/langs](https://github.com/ast-grep/langs?tab=readme-ov-file#packages) repository.

### Main functions

You can use `parse` to transform a string to ast-grep's main object `SgRoot`.
ast-grep also provides other utility for parse kind string and construct pattern.

```ts
/** Parse a string to an ast-grep instance */
export function parse(lang: Lang, src: string): SgRoot
/** Get the `kind` number from its string name. */
export function kind(lang: Lang, kindName: string): number
/** Compile a string to ast-grep Pattern. */
export function pattern(lang: Lang, pattern: string): NapiConfig
```

#### Example

```ts
import { parse, Lang } from '@ast-grep/napi'

const ast = parse(Lang.JavaScript, source)
const root = ast.root()
root.find("console.log")
```

### SgRoot

You will get an `SgRoot` instance when you `parse(lang, string)`.

`SgRoot` can also be accessed in `lang.findInFiles`'s callback by calling `node.getRoot()`.

In the latter case, `sgRoot.filename()` will return the path of the matched file.

#### Type

```ts
/** Represents the parsed tree of code. */
class SgRoot {
  /** Returns the root SgNode of the ast-grep instance. */
  root(): SgNode
  /**
   * Returns the path of the file if it is discovered by ast-grep's `findInFiles`.
   * Returns `"anonymous"` if the instance is created by `parse(lang, source)`.
   */
  filename(): string
}
```

#### Example

```ts
import { parse, Lang } from '@ast-grep/napi'

const ast = parse(Lang.JavaScript, source)
const root = ast.root()
root.find("console.log")
```

### SgNode

The main interface to traverse the AST.

#### Type

Most methods are self-explanatory. Please submit a new [issue](https://github.com/ast-grep/ast-grep/issues/new/choose) if you find something confusing.

```ts
class SgNode {
  // Read node's information
  range(): Range
  isLeaf(): boolean
  isNamed(): boolean
  isNamedLeaf(): boolean
  kind(): string
  // check if node has kind
  is(kind: string): boolean
  // for TypeScript type narrow
  kindToRefine: string
  text(): string
  // Check if node meets certain patterns
  matches(m: string): boolean
  inside(m: string): boolean
  has(m: string): boolean
  precedes(m: string): boolean
  follows(m: string): boolean
  // Get nodes' matched meta variables
  getMatch(m: string): SgNode | null
  getMultipleMatches(m: string): Array<SgNode>
  // Get node's SgRoot
  getRoot(): SgRoot
  // Traverse node tree
  children(): Array<SgNode>
  find(matcher: string | number | NapiConfig): SgNode | null
  findAll(matcher: string | number | NapiConfig): Array<SgNode>
  field(name: string): SgNode | null
  parent(): SgNode | null
  child(nth: number): SgNode | null
  ancestors(): Array<SgNode>
  next(): SgNode | null
  nextAll(): Array<SgNode>
  prev(): SgNode | null
  prevAll(): Array<SgNode>
  // Edit
  replace(text: string): Edit
  commitEdits(edits: Edit[]): string
}
```

Some methods have more sophisticated type signatures for the ease of use. See the [source code](https://github.com/ast-grep/ast-grep/blob/0999cdb542ff4431e3734dad38fcd648de972e6a/crates/napi/types/sgnode.d.ts#L38-L41) and our [tech blog](/blog/typed-napi.html)

### NapiConfig

`NapiConfig` is used in `find` or `findAll`.

#### Type

`NapiConfig` has similar fields as the [rule config](/reference/yaml.html).

```ts
interface NapiConfig {
  rule: object
  constraints?: object
  language?: FrontEndLanguage
  transform?: object
  utils?: object
}
```

### FindConfig

`FindConfig` is used in `findInFiles`.

#### Type

```ts
interface FindConfig {
  // You can search multiple paths
  // ast-grep will recursively find all files under the paths.
  paths: Array<string>
  // Specify what nodes will be matched
  matcher: NapiConfig
}
```

### Edit

`Edit` is used in `replace` and `commitEdits`.

```ts
interface Edit {
  startPos: number
  endPos: number
  insertedText: string
}
```

### Useful Examples

* [Test Case Source](https://github.com/ast-grep/ast-grep/blob/main/crates/napi/__test__/index.spec.ts) for `@ast-grep/napi`
* ast-grep usage in [vue-vine](https://github.com/vue-vine/vue-vine/blob/b661fd2dfb54f2945e7bf5f3691443e05a1ab8f8/packages/compiler/src/analyze.ts#L32)

### Language Object (deprecated)&#x20;

:::details language objects are deprecated

`ast-grep/napi` also has special language objects for `html`, `js` and `css`. They are deprecated and will be removed in the next version.

A language object has following methods.

```ts
/**
 * @deprecated language specific objects are deprecated
 * use the equivalent functions like `parse` in @ast-grep/napi
 */
export declare namespace js {
  /** @deprecated use `parse(Lang.JavaScript, src)` instead */
  export function parse(src: string): SgRoot
  /** @deprecated use `parseAsync(Lang.JavaScript, src)` instead */
  export function parseAsync(src: string): Promise<SgRoot>
  /** @deprecated use `kind(Lang.JavaScript, kindName)` instead */
  export function kind(kindName: string): number
  /** @deprecated use `pattern(Lang.JavaScript, p)` instead */
  export function pattern(pattern: string): NapiConfig
  /** @deprecated use `findInFiles(Lang.JavaScript, config, callback)` instead */
  export function findInFiles(
    config: FindConfig,
    callback: (err: null | Error, result: SgNode[]) => void
  ): Promise<number>
}
```

#### Example

```ts
import { js } from '@ast-grep/napi'

const source = `console.log("hello world")`
const ast = js.parse(source)
```

:::

## Python API

### SgRoot

The entry point object of ast-grep. You can use SgRoot to parse a string into a syntax tree.

```python
class SgRoot:
    def __init__(self, src: str, language: str) -> None: ...
    def root(self) -> SgNode: ...
```

### SgNode

Most methods are self-explanatory. Please submit a new [issue](https://github.com/ast-grep/ast-grep/issues/new/choose) if you find something confusing.

```python
class SgNode:
    # Node Inspection
    def range(self) -> Range: ...
    def is_leaf(self) -> bool: ...
    def is_named(self) -> bool: ...
    def is_named_leaf(self) -> bool: ...
    def kind(self) -> str: ...
    def text(self) -> str: ...

    # Refinement
    def matches(self, **rule: Unpack[Rule]) -> bool: ...
    def inside(self, **rule: Unpack[Rule]) -> bool: ...
    def has(self, **rule: Unpack[Rule]) -> bool: ...
    def precedes(self, **rule: Unpack[Rule]) -> bool: ...
    def follows(self, **rule: Unpack[Rule]) -> bool: ...
    def get_match(self, meta_var: str) -> Optional[SgNode]: ...
    def get_multiple_matches(self, meta_var: str) -> List[SgNode]: ...
    def get_transformed(self, meta_var: str) -> Optional[str]: ...
    def __getitem__(self, meta_var: str) -> SgNode: ...

    # Search
    @overload
    def find(self, config: Config) -> Optional[SgNode]: ...
    @overload
    def find(self, **kwargs: Unpack[Rule]) -> Optional[SgNode]: ...
    @overload
    def find_all(self, config: Config) -> List[SgNode]: ...
    @overload
    def find_all(self, **kwargs: Unpack[Rule]) -> List[SgNode]: ...

    # Tree Traversal
    def get_root(self) -> SgRoot: ...
    def field(self, name: str) -> Optional[SgNode]: ...
    def parent(self) -> Optional[SgNode]: ...
    def child(self, nth: int) -> Optional[SgNode]: ...
    def children(self) -> List[SgNode]: ...
    def ancestors(self) -> List[SgNode]: ...
    def next(self) -> Optional[SgNode]: ...
    def next_all(self) -> List[SgNode]: ...
    def prev(self) -> Optional[SgNode]: ...
    def prev_all(self) -> List[SgNode]: ...

    # Edit
    def replace(self, new_text: str) -> Edit: ...
    def commit_edits(self, edits: List[Edit]) -> str: ...
```

### Rule

The `Rule` object is a Python representation of the [YAML rule object](/guide/rule-config/atomic-rule.html) in the CLI. See the [reference](/reference/rule.html).

```python
class Pattern(TypedDict):
    selector: str
    context: str

class Rule(TypedDict, total=False):
    # atomic rule
    pattern: str | Pattern
    kind: str
    regex: str

    # relational rule
    inside: Relation
    has: Relation
    precedes: Relation
    follows: Relation

    # composite rule
    all: List[Rule]
    any: List[Rule]
    # pseudo code below for demo.
    "not": Rule # Python does not allow "not" keyword as attribute
    matches: str

# Relational Rule Related
StopBy = Union[Literal["neighbor"], Literal["end"], Rule]
class Relation(Rule, total=False):
    stopBy: StopBy
    field: str
```

### Config

The Config object is similar to the [YAML rule config](/guide/rule-config.html) in the CLI. See the [reference](/reference/yaml.html).

```python
class Config(TypedDict, total=False):
    rule: Rule
    constraints: Dict[str, Mapping]
    utils: Dict[str, Rule]
    transform: Dict[str, Mapping]
```

### Edit

`Edit` is used in `replace` and `commitEdits`.

```python
class Edit:
    # The start position of the edit
    start_pos: int
    # The end position of the edit
    end_pos: int
    # The text to be inserted
    inserted_text: str
```

## Rust API

Rust API is not stable yet. The following link is only for those who are interested in modifying ast-grep's source.

https://docs.rs/ast-grep-core/latest/ast\_grep\_core/

---

---
url: /reference/languages.md
---
# List of Languages with Built-in Support

The table below lists all languages that are supported by ast-grep.

**Alias** is the name you can use as an argument in `ast-grep run --lang [alias]` or as a value in YAML rule with `language: [alias]`.

**Extension** specifies the file extensions that ast-grep will look for when scanning the file system. By default, ast-grep uses the file extensions to determine the language.

***

| Language Name | Alias | File Extension |
|---|---|---|
|Bash | `bash` | `bash`, `bats`, `cgi`, `command`, `env`, `fcgi`, `ksh`, `sh`, `sh.in`, `tmux`, `tool`, `zsh` |
|C | `c` | `c`,`h`|
|Cpp | `cc`, `c++`, `cpp`, `cxx` | `cc`, `hpp`, `cpp`, `c++`, `hh`, `cxx`, `cu`, `ino`|
|CSharp | `cs`, `csharp` | `cs`|
|Css | `css` | `css`|
|Elixir | `ex`, `elixir` | `ex`, `exs`|
|Go | `go`, `golang` | `go`|
|Haskell | `hs`, `haskell` | `hs`|
|Hcl | `hcl` | `hcl`|
|Html | `html` | `html`, `htm`, `xhtml`|
|Java | `java` | `java`|
|JavaScript | `javascript`, `js`, `jsx` | `cjs`, `js`, `mjs`, `jsx`|
|Json | `json` | `json` |
|Kotlin | `kotlin`, `kt` | `kt`, `ktm`, `kts`|
|Lua | `lua` | `lua`|
|Nix | `nix` | `nix`|
|Php | `php` | `php` |
|Python | `py`, `python` | `py`, `py3`, `pyi`, `bzl`|
|Ruby | `rb`, `ruby` | `rb`, `rbw`, `gemspec`|
|Rust | `rs`, `rust` | `rs`|
|Scala | `scala` | `scala`, `sc`, `sbt`|
|Solidity | `solidity`, `sol` | `sol`|
|Swift | `swift` | `swift`|
|TypeScript | `ts`, `typescript` | `ts`, `cts`, `mts`|
|Tsx | `tsx` | `tsx`|
|Yaml | `yml` | `yml`, `yaml`|

***

:::tip Pro Tips
You can use [`languageGlobs`](/reference/sgconfig.html#languageglobs) to customize languages' extension mapping.
:::

---

---
url: /reference/playground.md
---
# ast-grep Playground Manual

The [ast-grep playground](/playground.html) is an online tool that allows you to try out ast-grep without installing anything on your machine. You can write code patterns and see how they match your code in real time. You can also apply rewrite rules to modify your code based on the patterns.

See the video for a quick overview of the playground.

The playground is a great way to *learn* ast-grep, *debug* patterns/rules, *report bugs* and *showcase* ast-grep's capabilities.

## Basic Usage

Annotated screenshot of the ast-grep playground:

![ast-grep playground](https://user-images.githubusercontent.com/2883231/268551825-2adfe739-c3d1-48c3-94d7-3c0c40fabbbc.png)

The ast-grep playground has a simple and intuitive layout that consists of four main areas.

### 1. Source Editor

The **source editor** is where you can write or paste the code that you want to search or modify.  The source editor supports syntax highlighting and auto-indentation for various languages, such as Python, JavaScript, Java, C#, and more.

:::tip How to Change Language?
You can choose the language of your code from the drop-down menu at the top right corner.
:::

### 2. Source AST Dump

The **source AST dump** is where you can see the AST representation of your source code. The AST dump shows the structure and the [kind and field](/advanced/core-concepts.html#kind-vs-field) of each node in the AST. You can use the AST dump to understand how your code is parsed and how to write patterns that match specific nodes or subtrees.

### 3. Matcher Editor

The **matcher editor** is where you can write the code patterns and rewrite rules that you want to apply to your source code. The matcher uses the same language as your source code. The matcher editor has two tabs: **Pattern** and **YAML**.

* **Pattern** provides an *approachable* option where you can write the [code pattern](/guide/pattern-syntax.html) that you want to match in your source code. You can also write a rewrite expression that specifies how to modify the matched code in the subeditor below. It roughly emulates the behavior of [`ast-grep run`](/reference/cli/run.html).
* **YAML** provides an *advanced* option where you can write a [YAML rule](/reference/yaml.html) that defines the pattern and metadata for your ast-grep scan. You can specify the [rule object](/reference/rule.html), id, message, severity, and other options for your rule. It is a web counterpart of [`ast-grep scan`](/reference/cli/scan.html).

### 4. Matcher Info

The **matcher info** is where you can see the information for the matcher section. The matcher info shows different information depending on which tab you are using in the matcher editor: **Pattern** or **YAML**.

* If you are using the **Pattern** tab, the matcher info shows the AST dump of your code pattern like the source AST dump.
* If you are using the **YAML** tab, the matcher info shows the matched meta-variables and errors if your rule is not valid. You can use the matched meta-variables to see which nodes in the source AST are bound to which variables in your pattern and rewrite expression. You can also use the errors to fix any issues in your rule.

***

#### YAML Tab Screenshot

![YAML](https://user-images.githubusercontent.com/2883231/268738518-279f0635-d5af-4b41-87c6-4bd6fa67b135.png)

## Share Results

In addition to the four main areas, the playground also has a **share button** at the bottom right corner. You can use this button to generate a unique URL that contains your source code, patterns, rules, and language settings. You can copy this URL and share it with others who want to try out your ast-grep session.

## View Diffs

Another feature of the ast-grep playground is the **View Diffs** option. You can use this option to see how your source code is changed by your rewrite expression or the [`fix`](/reference/yaml.html#fix) option in your YAML rule.

You can access this option by clicking the **Diff** tab in the source editor area. The Diff tab will show you a unified inline comparison of your original code and your modified code.

![Diff Tab Illustration](https://user-images.githubusercontent.com/2883231/268726696-d5091342-bc07-4859-8c95-abf079221cc2.png)

This is a useful way to check and debug your rule/pattern before applying it to your code base.

## Toggle Full AST Display

Sometimes you need to match code based on elements that are not encoded in AST. These elements are called [unnamed nodes](/advanced/core-concepts.html#named-vs-unnamed) in ast-grep.

ast-grep can represent code using two different types of tree structures: **AST** and **CST**.
**AST**, Abstract Syntax Tree, is a simplified representation of the code *excluding* unnamed nodes. **CST**, Concrete Syntax Tree, is a more detailed representation of the code *including* unnamed nodes. We have a standalone [doc page](/advanced/core-concepts.html#ast-vs-cst) for a deep-dive explanation of the two concepts.

In case you need to match unnamed nodes, you can toggle between AST and CST in the ast dumper by clicking the **Show Full Tree** option. This option will show you the full CST of your code, which may be useful for debugging or fine-tuning your patterns and rules.

|Syntax Tree Format|Screenshot|
|---|---|
|Named AST|![no full](https://user-images.githubusercontent.com/2883231/268730796-57ffb3be-e2e9-4199-8a71-76f1320cebf7.png)|
|Full CST|![full tree](https://user-images.githubusercontent.com/2883231/268730525-ea3b7c71-5389-42e5-abee-fc0d845e4b1b.png)|

## Test Multiple Rules

One of the cool features of the ast-grep playground is that you can test multiple rules at once! This can help you simulate how ast-grep would work in your real projects, where you might have several rules to apply to your code base.

To test multiple rules, you just need to separate them by `---` in the YAML editor. Each rule will have its own metadata and options, and you can see the results of each rule in the Source tab as well as the Diff tab.

Example with [playground link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiIyBhc3QtZ3JlcCBub3cgc3VwcG9ydHMgbXVsdGlwbGUgcnVsZXMgaW4gcGxheWdyb3VuZCFcbnJ1bGU6XG4gIHBhdHRlcm46IGNvbnNvbGUubG9nKCRBKVxuZml4OlxuICBsb2dnZXIubG9nKCRBKVxuLS0tXG5ydWxlOlxuICBwYXR0ZXJuOiBmdW5jdGlvbiAkQSgpIHsgJCQkQk9EWSB9XG5maXg6ICdjb25zdCAkQSA9ICgpID0+IHsgJCQkQk9EWSB9JyIsInNvdXJjZSI6Ii8vIGNvbnNvbGUubG9nKCkgd2lsbCBiZSBtYXRjaGVkIGJ5IHBhdHRlcm4hXG4vLyBjbGljayBkaWZmIHRhYiB0byBzZWUgcmV3cml0ZS5cblxuZnVuY3Rpb24gdHJ5QXN0R3JlcCgpIHtcbiAgY29uc29sZS5sb2coJ21hdGNoZWQgaW4gbWV0YXZhciEnKVxufVxuXG5jb25zdCBtdWx0aUxpbmVFeHByZXNzaW9uID1cbiAgY29uc29sZVxuICAgLmxvZygnQWxzbyBtYXRjaGVkIScpIn0=):

```yaml
rule:
  pattern: console.log($A)
fix:
  logger.log($A)
---
rule:
  pattern: function $A() { $$$BODY }
fix: 'const $A = () => { $$$BODY }'
```

Screenshot:

![multiple rule](https://user-images.githubusercontent.com/2883231/268735920-e6369832-6fa9-4b64-8975-2e813dc14076.png)

## Test Rule Diagnostics

Finally, the ast-grep playground also has a powerful feature that lets you see how your YAML rule reports diagnostics in the code editor.

This feature is optional, but can be turned on easily. To enable it, you need to specify the following fields in your YAML rule: `id`, `message`, `rule`, and `severity`. The `severity` field should be either `error`, `warning` or `info`, but not `hint`.

The playground will then display the diagnostics in the code editor with red or yellow wavy underlines, depending on the severity level. You can also hover over the underlines to see the message and the rule id for each diagnostic. This feature can help you detect and correct code issues more quickly and effectively.

[Example Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiaWQ6IG5vLWNvbnNvbGVcbnJ1bGU6XG4gIHBhdHRlcm46IGNvbnNvbGUuJE1FVEhPRCgkQSlcbm1lc3NhZ2U6IFVuZXhwZWN0ZWQgY29uc29sZVxuc2V2ZXJpdHk6IHdhcm5pbmdcblxuLS0tXG5cbmlkOiBuby1kZWJ1Z2dlclxucnVsZTpcbiAgcGF0dGVybjogZGVidWdnZXJcbm1lc3NhZ2U6IFVuZXhwZWN0ZWQgZGVidWdnZXJcbnNldmVyaXR5OiBlcnJvciIsInNvdXJjZSI6ImZ1bmN0aW9uIHRyeUFzdEdyZXAoKSB7XG4gIGNvbnNvbGUubG9nKCdtYXRjaGVkIGluIG1ldGF2YXIhJylcbn1cblxuY29uc3QgbXVsdGlMaW5lRXhwcmVzc2lvbiA9XG4gIGNvbnNvbGVcbiAgIC5sb2coJ0Fsc28gbWF0Y2hlZCEnKVxuXG5pZiAodHJ1ZSkge1xuICBkZWJ1Z2dlclxufSJ9)

![diagnostics](https://user-images.githubusercontent.com/2883231/268741624-98017dd4-8093-4b11-aa6f-cf7b66e68762.png)

---

---
url: /advanced/prompting.md
---
# Using ast-grep with AI Tools

This guide outlines several existing methods for leveraging AI with ast-grep.

:::warning Disclaimer
The field of AI is constantly evolving. The approaches detailed here are for reference, and we encourage you to explore and discover the best ways to utilize ast-grep with emerging AI technologies.
:::

## Using ast-grep with Claude Code Skill

This skill teaches Claude how to write and use ast-grep rules to perform advanced code searches. Unlike traditional text-based search (grep, ripgrep), ast-grep understands the structure of your code, allowing you to find patterns like:

* "Find all async functions that don't have error handling"
* "Locate all React components that use a specific hook"
* "Find functions with more than 3 parameters"
* "Search for console.log calls inside class methods"

Clone or download the [ast-grep skill repository](https://github.com/ast-grep/claude-skill) to your Claude Code skills directory:

```bash
# If you have a skills directory configured
cp -r ast-grep ~/.claude/skills/

# Or place it wherever your Claude Code skills are located
```

The skill should be automatically detected by Claude Code. You can verify by checking available skills in Claude Code.

## Simple Prompting in `AGENTS.md`

For everyday development, you can instruct your AI agent to use ast-grep for code searching and analysis. This method is straightforward but requires a model with up-to-date knowledge of ast-grep to be effective. If the model is not familiar with the tool, it may not utilize it as instructed.

You can set a system-level prompt for your AI agent to prioritize ast-grep for syntax-aware searches. Here is an example prompt comes from [this social post](https://x.com/kieranklaassen/status/1938453871086682232).

**Example Prompt:**

> You are operating in an environment where `ast-grep` is installed. For any code search that requires understanding of syntax or code structure, you should default to using `ast-grep --lang [language] -p '<pattern>'`. Adjust the `--lang` flag as needed for the specific programming language. Avoid using text-only search tools unless a plain-text search is explicitly requested.

This approach is best suited for general code queries and explorations within your projects.

## Providing Comprehensive Context to LLMs

Large Language Models (LLMs) with extensive context windows can be made highly effective at using ast-grep by providing them with its complete documentation.

The `llms.txt` file for ast-grep is a compilation of the entire documentation, designed to be fed into an LLM's context. This method significantly reduces the likelihood of the model "hallucinating" or generating incorrect ast-grep rules by giving it a thorough and accurate knowledge base to draw from.

You can find the full `llms.txt` file here: <https://ast-grep.github.io/llms-full.txt>

By loading this text into your session with a capable LLM, you can ask more complex questions and receive more accurate and nuanced answers regarding ast-grep's features and usage.

## Advanced Rule Development with MCP and Sub-agents

For more sophisticated and dedicated code analysis tasks, you can use the ast-grep Model Context Protocol (MCP) server. The [ast-grep-mcp](https://github.com/ast-grep/ast-grep-mcp) is an experimental server that connects AI assistants, such as Cursor and Claude Code, with the powerful structural search capabilities of ast-grep. This allows the AI to interact with your codebase in a more structured and intelligent way, moving beyond simple text-based searches.

The MCP server provides a set of tools that enable an AI to develop and refine ast-grep rules through a process of trial and error. This is particularly useful for creating complex rules that may require several iterations to perfect.

The core of this approach is to have the AI follow a systematic process for rule development:

```
## Rule Development Process
1. Break down the user's query into smaller parts.
2. Identify sub rules that can be used to match the code.
3. Combine the sub rules into a single rule using relational rules or composite rules.
4. if rule does not match example code, revise the rule by removing some sub rules and debugging unmatching parts.
5. Use ast-grep mcp tool to dump AST or dump pattern query
6. Use ast-grep mcp tool to test the rule against the example code snippet.
```

This iterative process allows the AI to "think" more like a human developer, refining its approach until the rule is correct. You can view a detailed prompt for this agentic rule development process in the `ast-grep-mcp` repository: <https://github.com/ast-grep/ast-grep-mcp/blob/main/ast-grep.mdc>.

---

---
url: /advanced/faq.md
---
# Frequently Asked Questions

## My pattern does not work, why?

1. **Use the Playground**: Test your pattern in the [ast-grep playground](/playground.html).
2. **Check for Valid Code**: Make sure your pattern is valid code that tree-sitter can parse.
3. **Ensure Correctness**: Use a [pattern object](/guide/rule-config/atomic-rule.html#pattern) to ensure your code is correct and unambiguous.
4. **Explore Examples**: See ast-grep's [catalog](/catalog/) for more examples.

The most common scenario is that you only want to match a sub-expression or one specific AST node in a whole syntax tree.
However, the code fragment corresponding to the sub-expression may not be valid code.
To make the code can be parsed by tree-sitter, you probably need more context instead of providing just code fragment.

For example, if you want to match key-value pair in JSON, writing `"key": "$VAL"` will not work because it is not a legal JSON.

Instead, you can provide context via the pattern object. See [playground code](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6Impzb24iLCJxdWVyeSI6ImZvbygkJCRBLCBiLCAkJCRDKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAne1widmVyc2lvblwiOiBcIiRWRVJcIiB9J1xuICAgIHNlbGVjdG9yOiBwYWlyIiwic291cmNlIjoie1xuICAgIFwidmVyc2lvblwiOiBcInZlclwiXG59In0=).

```YAML
rule:
  pattern:
    context: '{"key": "$VAL"}'
    selector: pair
```

The idea is that you can write full an valid code in the `context` field and use `selector` to select the sub-AST node.

This trick can be used in other languages as well, like [C](/catalog/c/#match-function-call) and [Go](/catalog/go/#match-function-call-in-golang). That said, pattern is not always the best choice for code search. [Rule](/guide/rule-config.html) can be more expressive and powerful.

## My Rule does not work, why?

Here are some tips to debug your rule:

* Use the [ast-grep playground](/playground.html) to test your rule.
* Simplify your rule to the minimal possible code that reproduces the issue.
* Confirm pattern's matched AST nodes are expected. e.g. statement and expression are [different matches](/advanced/pattern-parse.html#extract-effective-ast-for-pattern). This usually happens when you use `follows` or `precedes` in the rule.
* Check the [rule order](/advanced/faq.html#why-is-rule-matching-order-sensitive). The order of rules matters in ast-grep especially when using meta variables with relational rules.

## CLI and Playground produce different results, why?

There are two main reasons why the results may differ:

* **Parser Version**: The CLI may use a different version of the tree-sitter parser than the Playground.
  Playground parsers are updated less frequently than the CLI, so there may be differences in the results.
* **Text Encoding**: The CLI and Playground use different text encodings. CLI uses utf-8, while the Playground uses utf-16.
  The encoding difference may cause different fallback parsing during [error recovery](https://github.com/tree-sitter/tree-sitter/issues/224).

To debug the issue, you can use the [`--debug-query`](/reference/cli/run.html#debug-query-format) in the CLI to see the parsed AST nodes and meta variables.

```sh
ast-grep run -p <PATTERN> --debug-query ast
```

The debug output will show the parsed AST nodes and you can compare them with the [Playground](/playground.html). You can also use different debug formats like `cst` or `pattern`.

Different results are usually caused by incomplete or wrong code snippet in the pattern. A common fix is to provide a complete context code via the [pattern object](/reference/rule.html#atomic-rules).

```yaml
rule:
  pattern:
    context: 'int main() { return 0; }'
    selector: function
```

See [Pattern Deep Dive](/advanced/pattern-parse.html) for more context. Alternatively, you can try [rule](/guide/rule-config.html) instead.

Note `--debug-query` is not only for pattern, you can pass source code as `pattern` to see the parsed AST.

:::details Text encoding impacts tree-sitter error recovery.
Tree-sitter is a robust parser that can recover from syntax errors and continue parsing the rest of the code.
The exact strategy for error recovery is implementation-defined and uses a heuristic to determine the best recovery strategy.
See [tree-sitter issue](https://github.com/tree-sitter/tree-sitter/issues/224) for more details.

Text-encoding will affect the error recovery because it changed the cost of different recovery strategies.
:::

If you find the inconsistency between CLI and Playground, try confirming the playground version by hovering over the language label in playground, and the CLI version by [this file](https://github.com/ast-grep/ast-grep/blob/main/crates/language/Cargo.toml).

![Playground Version](/image/playground-parser-version.png)

:::tip Found inconsistency?
You can also [open an issue in the Playground repository](https://github.com/ast-grep/ast-grep.github.io/issues) if you find outdated parsers. Contribution to update the Playground parser is warmly welcome!
:::

## MetaVariable does not work, why?

1. **Correct Naming**: Start meta variables with the `$` sign, followed by uppercase letters (A-Z), underscores (`_`), or digits (1-9).
2. **Single AST Node**: A meta variable should be a single AST node. Avoid mixing meta variables with other text in one AST node. For example, `mix$OTHER_VAR` or `use$HOOK` will not work.
3. **Named AST Nodes**: By default, a meta variable matches only named AST nodes. Use double dollar signs like `$$UNNAMED` to match unnamed nodes.

## Multiple MetaVariable does not work

Multiple meta variables in ast-grep, such as `$$$MULTI`, are lazy. They stop matching nodes if the first node after them can match.

For example, `foo($$$A, b, $$$C)` matches `foo(a, c, b, b, c)`. `$$$A` stops before the first `b` and only matches `a, c`.

This design follows TypeScript's template literal types (`${infer VAR}Literal`) to ensure multiple meta variables always produce a match or non-match in linear time.

## Pattern cannot match my use case, how?

Patterns are a quick and easy way to match code in ast-grep, but they might not handle complex code. YAML rules are much more expressive and make it easier to specify complex code.

## I want to pattern match function call starts with some prefix string, how can I do that?

It is common to find function name or variable name following some naming convention like a function must starts with specific prefix.

For example, [React Hook](https://react.dev/learn/reusing-logic-with-custom-hooks#hook-names-always-start-with-use) in JavaScript requires function names start with `use`. Another example will be using `io_uring` in [Linux asynchronous programming](https://unixism.net/loti/genindex.html).

You may start with pattern like `use$HOOK` or `io_uring_$FUNC`. However, they are not valid meta variable names since the AST node text does not start with the dollar sign.

The workaround is using [`constraints`](https://ast-grep.github.io/guide/project/lint-rule.html#constraints) in [YAML rule](https://ast-grep.github.io/guide/project/lint-rule.html) and [`regex`](https://ast-grep.github.io/guide/rule-config/atomic-rule.html#regex) rule.

```yaml
rule:
  pattern: $HOOK($$$ARGS)
constraints:
  HOOK: { regex: '^use' }
```

[Example usage](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImZvbygkJCRBLCBiLCAkJCRDKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiAkSE9PSygkJCRBUkdTKVxuY29uc3RyYWludHM6XG4gIEhPT0s6IHsgcmVnZXg6IF51c2UgfSIsInNvdXJjZSI6ImZ1bmN0aW9uIFJlYWN0Q29tcG9uZW50KCkge1xuICAgIGNvbnN0IGRhdGEgPSBub3RIb28oKVxuICAgIGNvbnN0IFtmb28sIHNldEZvb10gPSB1c2VTdGF0ZSgnJylcbn0ifQ==).

:::danger MetaVariable must be one single AST node
Meta variables cannot be mixed with prefix/suffix string . `use$HOOK` and `io_uring_$FUNC` are not valid meta variables. They are parsed as one AST node as whole, and
ast-grep will not treat them as valid meta variable name.
:::

## How to reuse rule for similar languages like TS/JS or C/C++?

ast-grep does not support multiple languages in one rule because:

1. **Different ASTs**: Similar languages still have different ASTs. For instance, [JS](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiIiLCJzb3VyY2UiOiJmdW5jdGlvbiB0ZXN0KGEpIHt9In0=) and [TS](#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoidHlwZXNjcmlwdCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiIiLCJzb3VyY2UiOiJmdW5jdGlvbiB0ZXN0KGEpIHt9In0=) have different parsing trees for the same function declaration code.
2. **Different Kinds**: Similar languages may have different AST node kinds. Since ast-grep reports non-existing kinds as errors, there is no straightforward way to report error for kind only existing in one language.
3. **Debugging Experience**: Mixing languages in one rule requires users to test the rule in both languages. This can be confusing and error-prone, especially when unexpected results occur.

Supporting multi-lang rule is a challenging task for both tool developers and users. Instead, we recommend two approaches:

* **Always use the superset language**: Rule reusing usually happens when one language is a superset of another, e.g., TS and JS. In this case, you can use [`languageGlobs`](/reference/sgconfig.html#languageglobs) to parse files in the superset language. This is more suitable if you don't need to distinguish between the two languages.
* **Write Separate Rules**: Generate separate rules for each language. This approach is suitable when you do need to handle the differences between the languages.

If you have a better, clearer and easier proposal to support multi-lang rule, please leave a comment under [this issue](https://github.com/ast-grep/ast-grep/issues/525).

## Why is rule matching order sensitive?

ast-grep's rule matching is a step-by-step process. It matches one atomic rule at a time, stores the matched meta-variable, and proceeds to the next rule until all rules are matched.

**Rule matching is ordered** because previous rules' matched meta-variables can affect later rules. Only the first rule can specify what a `$META_VAR` matches, and later rules can only match the content captured by the first rule without modifying it.

Let's see an example. Suppose we want to find a recursive function in JavaScript. [This rule](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImZvbygkJCRBLCBiLCAkJCRDKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJpZDogcmVjdXJzaXZlLWNhbGxcbmxhbmd1YWdlOiBKYXZhU2NyaXB0XG5ydWxlOlxuICBhbGw6XG4gIC0gcGF0dGVybjogZnVuY3Rpb24gJEYoKSB7ICQkJCB9XG4gIC0gaGFzOlxuICAgICAgcGF0dGVybjogJEYoKVxuICAgICAgc3RvcEJ5OiBlbmRcbiIsInNvdXJjZSI6ImZ1bmN0aW9uIHJlY3Vyc2UoKSB7XG4gICAgZm9vKClcbiAgICByZWN1cnNlKClcbn0ifQ==) can do the trick.

:::code-group

```yml [rule.yml]
id: recursive-call
language: JavaScript
rule:
  all:
  - pattern: function $F() { $$$ }
  - has:
      pattern: $F()
      stopBy: end
```

```js [match.js]
function recurse() {
  foo()
  recurse()
}
```

:::

The rule works because the pattern `function $F() { $$$ }` matches first, capturing `$F` as `recurse`. The later `has` rule then looks for a `recurse()` call based on the matched `$F`.

If we [swap the order of rules](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImZvbygkJCRBLCBiLCAkJCRDKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJpZDogcmVjdXJzaXZlLWNhbGxcbmxhbmd1YWdlOiBKYXZhU2NyaXB0XG5ydWxlOlxuICBhbGw6XG4gIC0gaGFzOlxuICAgICAgcGF0dGVybjogJEYoKVxuICAgICAgc3RvcEJ5OiBlbmRcbiAgLSBwYXR0ZXJuOiBmdW5jdGlvbiAkRigpIHsgJCQkIH1cbiIsInNvdXJjZSI6ImZ1bmN0aW9uIHJlY3Vyc2UoKSB7XG4gICAgZm9vKClcbiAgICByZWN1cnNlKClcbn0ifQ==), it will produce no match.

```yml [rule.yml]
id: recursive-call
language: JavaScript
rule:
  all:
  - has:  # N.B. has is the first rule
      pattern: $F()
      stopBy: end
  - pattern: function $F() { $$$ }
```

In this case, the `has` rule matches first and captures `$F` as `foo` since `foo()` is the first function call matching the pattern `$F()`. The later rule `function $F() { $$$ }` will only find the `foo` declaration instead of `recurse`.

:::tip
Using `all` to specify the order of rule matching can be helpful when debugging YAML rules.
:::

## What does unordered rule object imply?

A rule object in ast-grep is an unordered dictionary. The order of rule application is implementation-defined. Currently, ast-grep applies atomic rules first, then composite rules, and finally relational rules.

If your rule depends on using meta variables in later rules, the best way is to use the `all` rule to specify the order of rules.

## `kind` and `pattern` rules are not working together, why?

The most common scenario is that your pattern is parsed as a different AST node than you expected. And you may use `kind` rule to filter out the AST node you want to match. This does not work in ast-grep for two reasons:

1. tree-sitter, the underlying parser library, does not offer a way to parse a string of a specific kind. So `kind` rule cannot be used to change the parsing outcome of a `pattern`.
2. ast-grep rules are mostly independent of each other, except sharing meta-variables during a match. `pattern` will behave the same regardless of another `kind` rule.

To specify the `kind` of a `pattern`, you need to use [pattern](/guide/rule-config/atomic-rule.html#pattern-object) [object](/advanced/pattern-parse.html#incomplete-pattern-code).

For example, to match class field in JavaScript, a kind + pattern rule [will not work](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IGEgPSAxMjNcbiAga2luZDogZmllbGRfZGVmaW5pdGlvbiIsInNvdXJjZSI6ImNsYXNzIEEge1xuICAgIGEgPSAxMjNcbn0ifQ==):

```yaml
# these are two separate rules
pattern: a = 123          # rule 1
kind: field_definition    # rule 2
```

This is because pattern `a = 123` is parsed as [`assignment_expression`](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiYSA9IDEyMyIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiIsInNvdXJjZSI6IiJ9). Pattern and kind are two separate rules. And using them together will match nothing because no AST will have both `assignment_expression` and `field_definition` kind at once.

Instead, you need to use pattern object to provide enough context code for the parser to parse the code snippet as `field_definition`:

```yaml
# this is one single pattern rule!
pattern:
  context: 'class A { a = 123 }' # provide full context code
  selector: field_definition     # select the effective pattern
```

Note the rule above is one single pattern rule, instead of two. The `context` field provides the full unambiguous code snippet of `class`. So the `a = 123` will be parsed as `field_definition`. The `selector` field then selects the `field_definition` node as the [effective pattern](/advanced/pattern-parse.html#steps-to-create-a-pattern) matcher.

## Does ast-grep support some advanced static analysis?

Short answer: **NO**.

Long answer: ast-grep at the moment does not support the following information:

* [scope analysis](https://eslint.org/docs/latest/extend/scope-manager-interface)
* [type information](https://semgrep.dev/docs/writing-rules/pattern-syntax#typed-metavariables)
* [control flow analysis](https://en.wikipedia.org/wiki/Control-flow_analysis)
* [data flow analysis](https://en.wikipedia.org/wiki/Data-flow_analysis)
* [taint analysis](https://semgrep.dev/docs/writing-rules/data-flow/taint-mode)
* [constant propagation](https://semgrep.dev/docs/writing-rules/data-flow/constant-propagation)

More concretely, it is not easy, or even possible, to achieve the following tasks in ast-grep:

* Find variables that are not defined/used in the current scope.
* Find variables of a specific type.
* Find code that is unreachable.
* Find code that is always executed.
* Identify the flow of user input.

Also see [tool comparison](/advanced/tool-comparison.html) for more information.

## I don't want to read the docs / I don't understand the docs / The docs are too long / I have an urgent request

[Open Source Software](https://antfu.me/posts/why-reproductions-are-required) is served "as-is" by volunteers. We appreciate your interest in ast-grep, but we also have limited time and resources to address every request.

We appreciate constructive feedback and are always looking for ways to improve the documentation and the tool itself. There are several ways you can help us or yourself:

* Ask [Copilot](https://copilot.microsoft.com/) or other AI assistants to help you understand the docs.
* Provide feedbacks or pull requests on the [documentation](https://github.com/ast-grep/ast-grep.github.io).
* Browse [Discord](https://discord.com/invite/4YZjf6htSQ), [StackOverflow](https://stackoverflow.com/questions/tagged/ast-grep) or [Reddit](https://www.reddit.com/r/astgrep/).

\~~If you just want an answer without effort, let the author [write a rule for you](https://github.com/sponsors/HerringtonDarkholme).~~

---

---
url: /advanced/how-ast-grep-works.md
---
# How ast-grep Works: A bird's-eye view

In the world of software development, efficiently searching, rewriting, linting, and analyzing code is essential for maintaining high-quality projects.

This is where **ast-grep** comes into play. Designed as a powerful structural search tool, ast-grep simplifies these tasks by leveraging the Abstract Syntax Tree (AST) representation of code. Let's break down how ast-grep works with the help of a diagram.

![Workflow](/image/diagram.png)

## The Workflow of ast-grep

Generally speaking, ast-grep takes user *queries of various input* formats, *parses the code into an AST* using TreeSitter, and performs *search, rewrite, lint, and analysis*,  utilizing the full power of CPU cores.

### **Query via Various Formats**

ast-grep can accept queries in multiple formats, making it flexible and user-friendly. Here are some common query formats:

* **Pattern Query**: Users can define [specific patterns](/guide/pattern-syntax.html) to search for within their codebase.
* **YAML Rule**: Structured rules written in [YAML](/guide/rule-config.html) format to guide the search and analysis process.
* **API Code**: Direct [API calls](/guide/api-usage.html) for more programmatic control over the searching and rewriting tasks.

### ast-grep's Core

ast-grep's core functionality is divided into two main components: parsing and matching.

#### 1. **Parsing with Tree-Sitter**

The core of ast-grep's functionality relies on **Tree-Sitter Parsers**. [TreeSitter](https://tree-sitter.github.io/) is a powerful parsing library that converts source code into an Abstract Syntax Tree (AST).
This tree structure represents the syntactic structure of the code, making it easier to analyze and manipulate.

#### 2. **Tree Matching**

Once the code is parsed into an AST, the ast-grep core takes over and finds the matching AST nodes based on the input queries.
Written in **Rust**, ast-grep ensures efficient performance by utilizing full CPU cores. This means it can handle large codebases and perform complex searches and transformations quickly.

### **Usage Scenarios**

ast-grep can be helpful for these scenarios.

* **Search**: Find specific patterns or constructs within the code.
* **Rewrite**: Automatically refactor or transform code based on predefined rules or patterns.
* **Lint**: Identify and report potential issues or code smells.
* **Analyze**: Perform in-depth code analysis to gather insights and metrics.

## Benefits of Using ast-grep

* **Multi-Core Processing**: ast-grep can handle multiple files in parallel by taking full advantage of multi-core processors. Typically ast-grep performs tasks faster than many other tools, making it suitable for large projects.
* **Versatility**: Whether you need to search for a specific code pattern, rewrite sections of code, lint for potential issues, or perform detailed analysis, ast-grep has you covered.

## Example in the Real World

* **Pattern + Search**: [CodeRabbit](https://coderabbit.ai/) uses ast-grep patterns to search code repo for code review knowledge.
  This example is collected from ast-grep's own [dogfooding](https://github.com/ast-grep/ast-grep/pull/780#discussion_r1425817237).

* **API + Rewrite**: [@vue-macros/cli](https://github.com/vue-macros/vue-macros-cli) is a CLI for rewriting at Vue Macros powered by ast-grep.

* **YAML + Lint**: [Vercel turbo](https://github.com/vercel/turbo/pull/8275) is using ast-grep to lint their Rust code with [custom rules](https://github.com/vercel/turbo/blob/main/.config/ast-grep/rules/no-context.yml).

## Conclusion

ast-grep is a versatile and efficient tool for modern software development needs. By parsing code into an Abstract Syntax Tree and leveraging the power of Rust, it provides robust capabilities for searching, rewriting, linting, and analyzing code. With multiple input formats and the ability to utilize full CPU cores, ast-grep is designed to handle the demands of today's complex codebases.

Whether you are maintaining a small project or a large enterprise codebase, ast-grep can help streamline your development workflow.

---

---
url: /advanced/core-concepts.md
---
# Core Concepts in ast-grep's Pattern

One key highlight of ast-grep is its pattern.

*Pattern is a convenient way to write and read expressions that describe syntax trees*. It resembles code, but with some special syntax and structure that allow you to match parts of a syntax tree based on their structure, type or content.

While ast-grep's pattern is **easy to learn**, it is **hard to master**. It requires you to know the Tree-sitter grammar and meaning of the target language, as well as the rules and conventions of ast-grep.

In this guide, we will help you grasp the core concepts of ast-grep's pattern that are common to all languages. We will also show you how to leverage the full power of ast-grep pattern for your own usage.

## What is Tree-sitter?

ast-grep is using [Tree-sitter](https://tree-sitter.github.io/) as its underlying parsing framework due to its **popularity**, **performance** and **robustness**.

Tree-sitter is a tool that generates parsers and provides an incremental parsing library.

A [parser](https://www.wikiwand.com/en/Parser_\(programming_language\)) is a program that takes a source code file as input and produces a tree structure that describes the organization of the code. (Contrary to ast-grep's name, the tree structure is not abstract syntax tree, as we will see later).

Writing good parsers for various programming languages is a laborious task, if even possible, for one single project like ast-grep. Fortunately, Tree-sitter is a venerable and popular tool that has a wide community support. Many mainstream languages such as C, Java, JavaScript, Python, Rust, and more are supported by Tree-sitter.
Using Tree-sitter as ast-grep's underlying parsing library allows it to *work with any language that has a well-maintained grammar available*.

Another perk of Tree-sitter is its incremental nature. An incremental parser is a parser that can update the syntax tree efficiently when the source code file is edited, without having to re-parse the entire file. *It can run very fast on every code changes in ast-greps' [interactive editing](https://ast-grep.github.io/guide/tooling-overview.html#interactive-mode).*

Finally, Tree-sitter also handles syntax errors gracefully, and it can parse multiple languages within the same file. *This makes pattern code more robust to parse and easier to write.* In future we can also support multi-language source code like Vue.

## Textual vs Structural

When you use ast-grep to search for patterns in source code, you need to understand the difference between textual and structural matching.

Source code input is text, a sequence of characters that follows certain syntax rules. You can use common search tools like [silver-searcher](https://github.com/ggreer/the_silver_searcher) or [ripgrep](https://github.com/BurntSushi/ripgrep) to search for text patterns in source code.

However, ast-grep does not match patterns against the text directly. Instead, it parses the text into a tree structure that represents the syntax of the code. This allows ast-grep to match patterns based on the structure of the code, not just its surface appearance. This is known as [structural](https://docs.sourcegraph.com/code_search/reference/structural) [search](https://docs.sourcegraph.com/code_search/reference/structural), which searches for code with a specific structure, not just a specific text.

*Therefore, the patterns you write must also be of valid syntax that can be compared with the code tree.*

:::tip Textual Search in ast-grep
Though `pattern` structurally matches code, you can use [the atomic rule `regex`](/guide/rule-config/atomic-rule.html#regex) to matches the text of a node by specifying a regular expression. This way, it is possible to combine textual and structural matching in ast-grep.
:::

## AST vs CST

To represent the syntax and structure of code, we have two types of tree structures: [AST](https://www.wikiwand.com/en/Abstract_syntax_tree) and [CST](https://eli.thegreenplace.net/2009/02/16/abstract-vs-concrete-syntax-trees/).

AST stands for Abstract Syntax Tree, which is a **simplified** representation of the code that *omits some details* like punctuation and whitespaces.
CST stands for Concrete Syntax Tree, which is a more **faithful** representation of the code that *includes all the details*.

Tree-sitter is a library that can parse code into CSTs for many programming languages. Thusly, *ast-grep, contrary to its name, searches and rewrites code based on CST patterns, instead of AST*.

Let's walk through an example to see why CST makes more sense.
Consider the JavaScript snippet `1 + 1`. Its AST representation [looks like this](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiY29uc29sZS5sb2coJE1BVENIKSIsImNvbmZpZyI6IiMgQ29uZmlndXJlIFJ1bGUgaW4gWUFNTFxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogaWYgKGZhbHNlKSB7ICQkJCB9XG4gICAgLSBwYXR0ZXJuOiBpZiAodHJ1ZSkgeyAkJCQgfVxuY29uc3RyYWludHM6XG4gICMgTUVUQV9WQVI6IHBhdHRlcm4iLCJzb3VyY2UiOiIxICsgMSJ9):

```
binary_expression
  number
  number
```

An astute reader should notice the important operator `+` is not encoded in AST. Meanwhile, its CST faithfully represents all critical information.

```
binary_expression
  number
  +                # note this + operator!
  number
```

You might wonder if using CST will make trivial whitespaces affect your search results.
Fortunately, ast-grep uses a [smart matching algorithm](/advanced/match-algorithm.html) that can skip trivial nodes in CST when appropriate, which saves you a lot of trouble.

## Named vs Unnamed

It is possible to convert CST to AST if we don't care about punctuation and whitespaces.
Tree-sitter has two types of nodes: named nodes and unnamed nodes(anonymous nodes).

The more important *named nodes* are defined with a regular name in the grammar rules, such as `binary_expression` or `identifier`. The less important *unnamed nodes* are defined with literal strings such as `","` or `"+"`.

Named nodes are more important for understanding the code's structure and meaning, while unnamed nodes are less important and can be sometimes skipped by ast-grep's matching algorithms.

The following example, adapted from [Tree-sitter's official guide](https://tree-sitter.github.io/tree-sitter/creating-parsers#the-first-few-rules), shows the difference in grammar definition.

```javascript
rules: {
  // named nodes are defined with the format `kind: parseRule`
  identifier: $ => /[a-z]+/,
  // binary_expression is also a named node,
  // the `+` operator is defined with a string literal, so it is an unnamed node
  binary_expression: $ => seq($.identifier, '+', $.identifier),
                                          // ↑ unnamed node
}
```

Practically, named nodes have a property called `kind` that indicates their names. You can use ast-grep's [atomic rule `kind`](/guide/rule-config/atomic-rule.html#kind) to find the specific AST node. [Playground link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJjb25maWciOiJydWxlOiBcbiAga2luZDogYmluYXJ5X2V4cHJlc3Npb24iLCJzb3VyY2UiOiIxICsgMSAifQ==) for the example below.

```yaml
rule:
  kind: binary_expression
# matches `1 + 1`
```

Further more, ast-grep's meta variable matches only named nodes by default. `return $A` matches only the first statement below. [Playground link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoicmV0dXJuICRBIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiIiwic291cmNlIjoicmV0dXJuIDEyM1xucmV0dXJuOyJ9).

```js
return 123 // `123` is named `number` and matched.
return;    // `;` is unnamed and not matched.
```

We can use double dollar `$$VAR` to *include unnamed nodes* in the pattern result. `return $$A` will match both statement above. [Playground link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoicmV0dXJuICQkQSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiIsInNvdXJjZSI6InJldHVybiAxMjNcbnJldHVybjsifQ==).

## Kind vs Field

Sometimes, using kind alone is not enough to find the nodes we want. A node may have several children with the same kind, but different roles in the code. For [example](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJjb25maWciOiJydWxlOlxuICBraW5kOiBzdHJpbmciLCJzb3VyY2UiOiJ2YXIgYSA9IHtcbiAgJ2tleSc6ICd2YWx1ZSdcbn0ifQ==), in JavaScript, an object may have multiple keys and values, all with the string kind.

To distinguish them, we can use `field` to specify the relation between a node and its parent. In ast-grep, `field` can be specified in two [relational rules](/guide/rule-config/relational-rule.html#relational-rule-mnemonics): `has` and `inside`.

`has` and `inside` accept a special configuration item called `field`. The value of `field` is the *field name* of the parent-child relation. For example, the key-value `pair` in JavaScript object has two children: one with field `key` and the other with field `value`. We can use [this rule](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJjb25maWciOiJydWxlOlxuICBraW5kOiBzdHJpbmdcbiAgaW5zaWRlOlxuICAgIGZpZWxkOiBrZXlcbiAgICBraW5kOiBwYWlyIiwic291cmNlIjoidmFyIGEgPSB7XG4gICdrZXknOiAndmFsdWUnXG59In0=) to match the `key` node of kind `string`.

```yaml
rule:
  kind: string
  inside:
    field: key
    kind: pair
```

`field` can help us to narrow down the search scope and make the pattern more precise.

We can also use `has` to rewrite the rule above, searching the key-value `pair` with `string` key. [Playground link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJjb25maWciOiJydWxlOlxuICBraW5kOiBwYWlyXG4gIGhhczpcbiAgICBmaWVsZDoga2V5XG4gICAga2luZDogc3RyaW5nIiwic291cmNlIjoidmFyIG1hdGNoID0geyAna2V5JzogJ3ZhbHVlJyB9XG52YXIgbm9NYXRjaCA9IHsga2V5OiB2YWx1ZX0ifQ==).

```yaml
rule:
  kind: pair
  has:
    field: key
    kind: string
```

:::tip Key Difference between `kind` and `field`

* `kind` is the property of the node itself. Only named nodes have `kind`s.
* `field` is the property of the relation between parent and child. Unnamed nodes can also have `field`s.
  :::

It might be confusing to new users that a node has both `kind` and `field`. `kind` belongs to the node itself, represented by blue text in ast-grep's playground. Child node has a `field` only relative to its parent, and vice-versa. `field` is represented by dark yellow text in the playground. Since field is a property of a node relation, unnamed nodes can also have `field`. For example, the `+` in the binary expression `1 + 1` has the field `operator`.

## Significant vs Trivial

ast-grep goes further beyond Tree-sitter. It has a concept about the "significance" of a node.

* If a node is a named node or has a field relative to its parent, it is a **significant** node.
* Otherwise, the node is a **trivial** node.

:::warning Even significance is not enough
Most Tree-sitter languages do not encode all critical structures in AST, the tree with named nodes only.
Even significant nodes are not sufficient to represent the meaning of code.
We have to preserve some trivial nodes for precise matching.
:::

Tree-sitter parsers do not encode all semantics with named nodes. For example, `class A { get method() {} }` and `class A { method() {} }` are equivalent in Tree-sitter's AST. The critical token `get` is not named nor has a field name. It is a trivial node!

If you do not care about if the method is a getter method, a static method or an instance method, you can use `class $A { method() {} }` to [match all the three methods at once](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiY2xhc3MgJEEgeyBtZXRob2QoKSB7fSB9IiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogcGFpclxuICBoYXM6XG4gICAgZmllbGQ6IGtleVxuICAgIGtpbmQ6IHN0cmluZyIsInNvdXJjZSI6ImNsYXNzIEEgeyBtZXRob2QoKSB7fX1cbmNsYXNzIEIgeyBnZXQgbWV0aG9kKCkge319XG5jbGFzcyBDIHsgc3RhdGljIG1ldGhvZCgpIHt9fSJ9). Alternatively, you can [fully spell out the method modifier](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiY2xhc3MgJEEgeyBnZXQgbWV0aG9kKCkge30gfSIsImNvbmZpZyI6InJ1bGU6XG4gIGtpbmQ6IHBhaXJcbiAgaGFzOlxuICAgIGZpZWxkOiBrZXlcbiAgICBraW5kOiBzdHJpbmciLCJzb3VyY2UiOiJjbGFzcyBBIHsgbWV0aG9kKCkge319XG5jbGFzcyBCIHsgZ2V0IG1ldGhvZCgpIHt9fVxuY2xhc3MgQyB7IHN0YXRpYyBtZXRob2QoKSB7fX0ifQ==) if you need to tell getter method from normal method.

## Summary

Thank you for reading until here! There are many concepts in this article. Let's summarize them in one paragraph.

ast-grep uses Tree-sitter to parse *textual* source code into a detailed tree *structure* called **CST**. We can get **AST** from CST by only keeping **named nodes**, which have kinds. To search nodes in a syntax tree, you can use both node **kind** and node **field**, which is a special role of a child node relative to its parent node. A node with either a kind or a field is a **significant** node.

---

---
url: /advanced/pattern-parse.md
---
# Deep Dive into ast-grep's Pattern Syntax

ast-grep's pattern is easy to learn but hard to master. While it's easy to get started with, mastering its nuances can greatly enhance your code searching capabilities.

This article aims to provide you with a deep understanding of how ast-grep's patterns are parsed, created, and effectively used in code matching.

## Steps to Create a Pattern

Parsing a pattern in ast-grep involves these keys steps:

1. Preprocess the pattern text, e.g, replacing `$` with [expando\_char](/advanced/custom-language.html#register-language-in-sgconfig-yml).
2. Parse the preprocessed pattern text into AST.
3. Extract effective AST nodes based on builtin heuristics or user provided [selector](/reference/rule.html#pattern).
4. Detect AST with wildcard text and convert them into [meta variables](/guide/pattern-syntax.html#meta-variable).

![image](/image/parse-pattern.jpg)

Let's dive deep into each of these steps.

## Pattern is AST based

***First and foremost, pattern is AST based**.*

ast-grep's pattern code will be converted into the Abstract Syntax Tree (AST) format, which is a tree structure that represents the code snippet you want to match.

Therefore pattern cannot be arbitrary text, but a valid code with meta variables as placeholders.
If the pattern cannot be parsed by the underlying parser tree-sitter, ast-grep won't be able to find valid matching for it.

There are several common pitfalls to avoid when creating patterns.

### Invalid Pattern Code

ast-grep pattern must be parsable valid code. While this may seem obvious, newcomers sometimes make mistakes when creating patterns with meta-variables.

***Meta-variable is usually parsed as identifier in most languages.***

When using meta-variables, make sure they are placed in a valid context and not used as a keyword or an operator.
For example, you may want to use `$OP` to match binary expressions like `a + b`.
The pattern below will not work because parsers see it as three consecutive identifiers separated by spaces.

```
$LEFT $OP $RIGHT
```

You can instead use [atomic rule](/guide/rule-config/atomic-rule.html#kind) `kind: binary_expression` to [match binary expressions](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIGtpbmQ6IGJpbmFyeV9leHByZXNzaW9uIiwic291cmNlIjoiYSArIGIgXHJcbmEgLSBiXHJcbmEgPT0gYiAifQ==).

Similarly, in JavaScript you may want to match [object accessors](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/Object_initializer#method_definitions) like `{ get foo() {}, set bar() { } }`.
The pattern below will not work because meta-variable is not parsed as the keywords `get` and `set`.

```js
obj = { $KIND foo() { } }
```

Again [rule](/guide/rule-config.html) is more suitable for [this scenario](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIGtpbmQ6IG1ldGhvZF9kZWZpbml0aW9uXG4gIHJlZ2V4OiAnXmdldHxzZXRcXHMnIiwic291cmNlIjoidmFyIGEgPSB7XHJcbiAgICBmb28oKSB7fVxyXG4gICAgZ2V0IGZvbygpIHt9LFxyXG4gICAgc2V0IGJhcigpIHt9LFxyXG59In0=).

```yaml
rule:
  kind: method_definition
  regex: '^get|set\s'
```

### Incomplete Pattern Code

It is very common and even attempting to write incomplete code snippet in patterns. However, incomplete code does not *always* work.

Consider the following JSON code snippet as pattern:

```json
"a": 123
```

While the intention here is clearly to match a key-value pair, tree-sitter does not treat it as valid JSON code because it is missing the enclosing `{}`. Consequently ast-grep will not be able to parse it.

The solution here is to use [pattern object](/guide/rule-config/atomic-rule.html#pattern-object) to provide complete code snippet.

```yaml
pattern:
  context: '{ "a": 123 }'
  selector: pair
```

You can use both ast-grep playground's [pattern tab](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoianNvbiIsInF1ZXJ5IjoieyBcImFcIjogMTIzIH0iLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiJwYWlyIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogbWV0aG9kX2RlZmluaXRpb25cbiAgcmVnZXg6ICdeZ2V0fHNldFxccyciLCJzb3VyY2UiOiJ7IFwiYVwiOiAxMjMgfSAifQ==) or [rule tab](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6Impzb24iLCJxdWVyeSI6InsgXCJhXCI6IDEyMyB9IiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoicGFpciIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IFxuICAgIGNvbnRleHQ6ICd7XCJhXCI6IDEyM30nXG4gICAgc2VsZWN0b3I6IHBhaXIiLCJzb3VyY2UiOiJ7IFwiYVwiOiAxMjMgfSAifQ==) to verify it.

***Incomplete pattern code sometimes works fine due to error-tolerance.***

For better *user experience*, ast-grep parse pattern code as lenient as possible. ast-grep parsers will try recovering parsing errors and ignoring missing language constructs.

For example, the pattern `foo(bar)` in Java cannot be parsed as valid code. However, ast-grep recover the parsing error, ignoring missing semicolon and treat it as a method call. So the pattern [still works](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YSIsInF1ZXJ5IjoiZm9vKGJhcikiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAne1wiYVwiOiAxMjN9J1xuICAgIHNlbGVjdG9yOiBwYWlyIiwic291cmNlIjoiY2xhc3MgQSB7XG4gICAgZm9vKCkge1xuICAgICAgICBmb28oYmFyKTtcbiAgICB9XG59In0=).

### Ambiguous Pattern Code

Just as programming languages have ambiguous grammar, so ast-grep patterns can be ambiguous.

Let's consider the JavaScript code snippet below:

```js
a: 123
```

It can be interpreted as an object key-value pair or a labeled statement.

Without other hints, ast-grep will parse it as labeled statement by default. To match object key-value pair, we need to provide more context by [using pattern object](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoieyBhOiAxMjMgfSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6InBhaXIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAne1wiYVwiOiAxMjN9J1xuICAgIHNlbGVjdG9yOiBwYWlyIiwic291cmNlIjoiYSA9IHsgYTogIDEyMyB9In0=).

```yaml
pattern:
  context: '{ a: 123 }'
  selector: pair
```

Other examples of ambiguous patterns include:

* Match function call in [Golang](/catalog/go/#match-function-call-in-golang) and [C](/catalog/c/#match-function-call)
* Match [class field](/guide/rule-config/atomic-rule.html#pattern-object) in JavaScript

### How ast-grep Handles Pattern Code?

ast-grep uses best efforts to parse pattern code for best user experience.

Here are some strategies ast-grep uses to handle code snippet:

* **Replace `$` with expando\_char**:
  some languages use `$` as a special character, so ast-grep replace it with [expando\_char](/advanced/custom-language.html#register-language-in-sgconfig-yml) in order to make the pattern code parsable.

* **Ignore missing nodes**: ast-grep will ignore missing nodes in pattern like trailing semicolon in Java/C/C++.

* **Treat root error as normal node**: if the parser error has no siblings, ast-grep will treat it as a normal node.

* If all above fails, users should provide more code via pattern object

:::warning Pattern Error Recovery is useful, but not guaranteed

ast-grep's recovery mechanism heavily depends on tree-sitter's behavior. We cannot guarantee invalid patterns will be parsed consistently between different versions. So using invalid pattern may lead to unexpected results after upgrading ast-grep.

When in doubt, always use valid code snippets with pattern object.
:::

## Extract Effective AST for Pattern

After parsing the pattern code, ast-grep needs to extract AST nodes to make the actual pattern.

Normally, a code snippet generated by tree-sitter will be a full AST tree. Yet it is unlikely that the entire tree will be used as a pattern. The code `123` will produce a tree like `program -> expression_statement -> number` in many languages. But we want to match a number literal in the code, not a program containing just a number.

ast-grep uses two strategies to extract **effective AST nodes** that will be used to match code.

### Builtin Heuristic

***By default, at-grep extracts the leaf node or the innermost node with more than one child.***

This heuristic extracts the most specific node while still keeping all structural information in the pattern.
If a node has only one child, it is atomic and cannot be further decomposed. We can safely assume the node contains no structural information for matching. In contrast, a node with more than one child contains a structure that we want to search.

Examples:

* `123` will be extracted as `number` because it is the leaf node.

```yaml
program
  expression_statement
    number              <--- effective node
```

See [Playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiMTIzIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiIiwic291cmNlIjoiIn0=).

* `foo(bar)` will be extracted as `call_expression` because it is the innermost node that has more than one child.

```yaml
program
  expression_statement
    call_expression       <--- effective node
      identifier
      arguments
        identifier
```

See [Playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiZm9vKGJhcikiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiJjYWxsX2V4cHJlc3Npb24iLCJjb25maWciOiIiLCJzb3VyY2UiOiIifQ==).

### User Defined Selector

Sometimes the effective node extracted by the builtin heuristic may not be what you want.
You can explicitly specify the node to extract using the [selector](/reference/rule.html#pattern) field in the rule configuration.

For example, you may want to match the whole `console.log` statement in JavaScript code. The effective node extracted by the builtin heuristic is `call_expression`, but you want to match the whole `expression_statement`.

Using `console.log($$$)` directly will not include the trailing `;` in the pattern, see [Playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiY29uc29sZS5sb2coJCQkKSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic2lnbmF0dXJlIiwic2VsZWN0b3IiOiJjYWxsX2V4cHJlc3Npb24iLCJjb25maWciOiIiLCJzb3VyY2UiOiJjb25zb2xlLmxvZyhmb28pXG5jb25zb2xlLmxvZyhiYXIpOyJ9).

```js
console.log("Hello")
console.log("World");
```

You can use pattern object to explicitly specify the effective node to be `expression_statement`. [Playground](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCQkJCkiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNpZ25hdHVyZSIsInNlbGVjdG9yIjoiY2FsbF9leHByZXNzaW9uIiwiY29uZmlnIjoicnVsZTpcbiAgcGF0dGVybjpcbiAgICBjb250ZXh0OiBjb25zb2xlLmxvZygkJCQpXG4gICAgc2VsZWN0b3I6IGV4cHJlc3Npb25fc3RhdGVtZW50XG5maXg6ICcnIiwic291cmNlIjoiY29uc29sZS5sb2coZm9vKVxuY29uc29sZS5sb2coYmFyKTsifQ==)

```yaml
pattern:
  context: console.log($$$)
  selector: expression_statement
```

Using `selector` is especially helpful when you are also using relational rules like `follows` and `precedes`.
You want to match the statement instead of the default inner expression node, and [match other statements around it](https://github.com/ast-grep/ast-grep/issues/1427).

:::tip
When in doubt, try pattern object first.
:::

## Meta Variable Deep Dive

ast-grep's meta variables are also AST based and are detected in the effective nodes extracted from the pattern code.

### Meta Variable Detection in Pattern

Not all `$` prefixed strings will be detected as meta variables.

Only AST nodes that match meta variable syntax will be detected.
If meta variable text is not the only text in the node or it spans multiple nodes, it will not be detected as a meta variable.

**Working meta variable examples:**

* `$A` works
  * `$A` is one single `identifier`
* `$A.$B` works
  * `$A` is `identifier` inside `member_expression`
  * `$B` is the `property_identifier`.
* `$A.method($B)` works
  * `$A` is `identifier` inside `member_expression`
  * `$B` is `identifier` inside `arguments`

**Non working meta variable examples:**

* `obj.on$EVENT` does not work
  * `on$EVENT` is `property_identifier` but `$EVENT` is not the only text
* `"Hello $WORLD"` does not work
  * `$WORLD` is inside `string_content` and is not the only text
* `a $OP b` does not work
  * the whole pattern does not parse
* `$jq` does not work
  * meta variable does not accept lower case letters

See all examples in [Playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzaWduYXR1cmUiLCJzZWxlY3RvciI6ImNhbGxfZXhwcmVzc2lvbiIsImNvbmZpZyI6IiIsInNvdXJjZSI6Ii8vIHdvcmtpbmdcbiRBXG4kQS4kQlxuJEEubWV0aG9kKCRCKVxuXG4vLyBub24gd29ya2luZ1xub2JqLm9uJEVWRU5UXG5cIkhlbGxvICRXT1JMRFwiXG5hICRPUCBiIn0=).

### Matching Unnamed Nodes

A meta variable pattern `$META` will capture [named nodes](/advanced/core-concepts.html#named-vs-unnamed) by default.
To capture [unnamed nodes](/advanced/core-concepts.html#named-vs-unnamed), you can use double dollar sign `$$VAR`.

Let's go back to the binary expression example. It is impossible to match arbitrary binary expression in one single pattern. But we can combine `kind` and `has` to match the operator in  binary expressions.

Note, `$OP` cannot match the operator because operator is not a named node. We need to use `$$OP` instead.

```yaml
rule:
  kind: binary_expression
  has:
    field: operator
    pattern: $$OP
    # pattern: $OP
```

See the above rule to match all arithmetic expressions in [action](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCQkJCkiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNpZ25hdHVyZSIsInNlbGVjdG9yIjoiY2FsbF9leHByZXNzaW9uIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogYmluYXJ5X2V4cHJlc3Npb25cbiAgaGFzOlxuICAgIGZpZWxkOiBvcGVyYXRvclxuICAgIHBhdHRlcm46ICQkT1BcbiAgICAjIHBhdHRlcm46ICRPUCIsInNvdXJjZSI6IjEgKyAxIn0=).

### How Multi Meta Variables Match Code

Multiple meta variables like `$$$ARGS` has special matching behavior. It will match multiple nodes in the AST.

`$$$ARGS` will match multiple nodes in source code when the meta variable starts to match. It will match as many nodes as possible until the first AST node after the meta var in pattern is matched.

The behavior is like [non-greedy](https://stackoverflow.com/questions/11898998/how-can-i-write-a-regex-which-matches-non-greedy) matching in regex and template string literal `infer` in [TypeScript](https://github.com/microsoft/TypeScript/pull/40336).

## Use ast-grep playground to debug pattern

ast-grep playground is a great tool to debug pattern code. The pattern tab and pattern panel can help you visualize the AST tree, effective nodes and meta variables.

![playground](/image/pattern-debugger.jpg)

In next article, we will explain how ast-grep's pattern is used to match code, the pattern matching algorithm.

---

---
url: /advanced/match-algorithm.md
---
# Deep Dive into ast-grep's Match Algorithm

By default, ast-grep uses a smart strategy to match pattern against the AST node. All nodes in the pattern must be matched, but it will skip unnamed nodes in target code.

For background and the definition of ***named*** and ***unnamed*** nodes, please refer to the [core concepts](/advanced/core-concepts.html) doc.

## How ast-grep's Smart Matching Works

Let's see an example in action.

The following pattern `function $A() {}` will match both plain function and async function in JavaScript. See [playground](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiZnVuY3Rpb24gJEEoKSB7fSIsInJld3JpdGUiOiJEZWJ1Zy5hc3NlcnQiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAneyAkTTogKCQkJEEpID0+ICRNQVRDSCB9J1xuICAgIHNlbGVjdG9yOiBwYWlyXG4iLCJzb3VyY2UiOiJmdW5jdGlvbiBhKCkge31cbmFzeW5jIGZ1bmN0aW9uIGEoKSB7fSJ9)

```js
// function $A() {}
function foo() {}    // matched
async function bar() {} // matched
```

This is because the keyword `async` is an unnamed node in the syntax tree, so the `async` in the code to search is skipped. As long as `function`, `$A` and `{}` are matched, the pattern is considered matched.

However, if the `async` keyword appears in the pattern code, it will [not be skipped](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiYXN5bmMgZnVuY3Rpb24gJEEoKSB7fSIsInJld3JpdGUiOiJ1c2luZyBuYW1lc3BhY2UgZm9vOjokQTsiLCJjb25maWciOiJcbmlkOiB0ZXN0YmFzZV9pbml0aWFsaXplclxubGFuZ3VhZ2U6IENQUFxucnVsZTpcbiAgcGF0dGVybjpcbiAgICBzZWxlY3RvcjogY29tcG91bmRfc3RhdGVtZW50XG4gICAgY29udGV4dDogXCJ7ICQkJEIgfVwiXG5maXg6IHwtXG4gIHtcbiAgICBmKCk7XG4gICAgJCQkQlxuICB9Iiwic291cmNlIjoiLy8gYXN5bmMgZnVuY3Rpb24gJEEoKSB7fVxuZnVuY3Rpb24gZm9vKCkge30gICAgLy8gbm90IG1hdGNoZWRcbmFzeW5jIGZ1bmN0aW9uIGJhcigpIHt9IC8vIG1hdGNoZWRcbiJ9) and is required to match node in the code.

```js
// async function $A() {}
function foo() {}    // not matched
async function bar() {} // matched
```

The design principle here is that the less a pattern specifies, the more code it can match. Every nodes the pattern author spells out will be respected by ast-grep's matching algorithm by default.

## Smart is Sometimes Dumb

The smart algorithm does not always behave as desired. There are cases where we need more flexibility in the matching algorithm. We may want to ignore all CST trivia nodes. Or even we want to ignore comment AST nodes.

Suppose we want to write a pattern to match import statement in JavaScript. The pattern `import $A from 'lib'` will match only `import A from 'lib'`, but not `import A from "lib"`. This is because the import string has different quotation marks. We do want to ignore the trivial unnamed nodes here.

To this end, ast-grep implements different pattern matching algorithms to provide more flexibility to the users, and every pattern can have their own matching algorithm to fine-tune the matching behavior.

## Matching Algorithm Strictness

Different matching algorithm is controlled by **pattern strictness**.

:::tip Strictness
Strictness is defined in terms of what nodes can be *skipped* during matching.

A *stricter* matching algorithm will *skip fewer nodes* and accordingly *produce fewer matches*.
:::

Currently, ast-grep has these strictness levels.

* `cst`: All nodes in the pattern and target code must be matched. No node is skipped.
* `smart`: All nodes in the pattern must be matched, but it will skip unnamed nodes in target code. This is the default behavior.
* `ast`: Only named AST nodes in both pattern and target code are matched. All unnamed nodes are skipped.
* `relaxed`: Named AST nodes in both pattern and target code are matched. Comments and unnamed nodes are ignored.
* `signature`: Only named AST nodes' kinds are matched. Comments, unnamed nodes and text are ignored.

## Strictness Examples

Let's see how strictness `ast` will impact matching. In our previous import lib example, the pattern `import $A from 'lib'` will match both two statements.

```js
import $A from 'lib' // pattern
import A1 from 'lib' // match, quotation is ignored
import A2 from "lib" // match, quotation is ignored
import A3 from "not" // no match, string_fragment is checked
```

First, the pattern and code will be parsed as the tree below. Named

The unnamed nodes are skipped during the matching. Nodes' namedness is annotated beside them.

```
import_statement    // named
  import            // unnamed
  import_clause     // named
    identifier      // named
  from              // unnamed
  string            // named
    "               // unnamed
    string_fragment // named
    "               // unnamed
```

Under the strictness of `ast`, the full syntax tree will be reduced to an Abstract Syntax Tree where only named nodes are kept.

```
import_statement
  import_clause
    identifier      // $A
  string
    string_fragment // lib
```

As long as the tree structure matches and the meta-variable `$A` and string\_fragment `lib` are matched, the pattern and code are counted as a match.

***

Another example will be matching the pattern `foo(bar)` across different strictness levels:

```ts
// exact match in all levels
foo(bar)
// match in all levels except cst due to the trailing comma in code
foo(bar,)
// match in relaxed and signature because comment is skipped
foo(/* comment */ bar)
// match in signature because text content is ignored
bar(baz)
```

## Strictness Table

Strictness considers both nodes' namedness and their locations, i.e,
*is the node named* and *is the node in pattern or code*

The table below summarize how nodes are skipped during matching.

|Strictness|Named Node in Pattern|Named Node in Code to Search|Unnamed Node in Pattern| Unnamed Node in Code to Search|
|---|----|---|---|---|
|`cst`| Keep | Keep| Keep | Keep |
|`smart`| Keep| Keep | Keep | Skip |
|`ast`| Keep| Keep | Skip| Skip |
|`relaxed`| Skip comment | Skip comment | Skip | Skip |
|`signature`| Skip comment. Ignore text | Skip comment. Ignore text | Skip | Skip |

## Configure Strictness

ast-grep has two ways to configure pattern strictness.

1. Using `--strictness` in `ast-grep run`

You can use the `--strictness` flag in [`ast-grep run`](/reference/cli/run.html)

```bash
ast-grep run -p '$FOO($BAR)' --strictness ast
```

2. Using `strictness` in Pattern Object

[Pattern object](/reference/rule.html#pattern) in YAML has an optional `strictness` field.

```
id: test-pattern-strictness
language: JavaScript
rule:
  pattern:
    context: $FOO($BAR)
    strictness: ast
```

---

---
url: /advanced/find-n-patch.md
---
# Find & Patch: A Novel Functional Programming like Code Rewrite Scheme

## Introduction

Code transformation is a powerful technique that allows you to modify your code programmatically. There are many tools that can help you with code transformation, such as [Babel](https://babeljs.io/)/[biome](https://github.com/biomejs/biome/discussions/1762) for JavaScript/TypeScript, [libcst](https://libcst.readthedocs.io/en/latest/) for Python, or [Rector](https://getrector.com/) for PHP. Most of these tools use imperative APIs to manipulate the [abstract syntax tree](https://www.wikiwand.com/en/Abstract_syntax_tree) (AST) of your code.

In this post, we will introduce a different approach to code transformation called **Find & Patch**.

This scheme lets you rewrite complex code using a fully declarative [Domain-Specific Language](https://www.wikiwand.com/en/Domain-specific_language) (DSL). While the scheme is powerful, the underlying concept is simple: find certain nodes, rewrite them, and recursively repeat the rewriting.

The idea of Find & Patch comes from developing [ast-grep](https://ast-grep.github.io/), a tool using AST to find and replace code patterns. We realized that this approach can be generalized and extended to support more complex and diverse code transformations!

At the end of this article, we will compare Find & Patch to functional programming on the tree of syntax nodes. You can apply filter nodes using `rule`, map them via `transform`, and compose them with `rewriters`.

This gives you a lot of flexibility and expressiveness to manipulate your code!

## What is ast-grep?

[ast-grep](https://github.com/ast-grep/ast-grep) is a tool to search and rewrite code based on ASTs. It is like `grep` for code, but with the power of ASTs.
More concretely, ast-grep can find code patterns using its [rule system](https://ast-grep.github.io/guide/rule-config/atomic-rule.html). It can also rewrite the matched code using [meta-variables](https://ast-grep.github.io/guide/pattern-syntax.html#meta-variable) based on the rule.

ast-grep's rewriting can be seen as two steps: finding target nodes and patching them with new text.

## Find and Patch: How ast-grep Rewrites Code

The basic rewriting workflow of ast-grep is like below:

1. *Find*: search the nodes in the AST that match the rewriter rules (hence the name ast-grep).
2. *Rewrite*: generate a new string based on the matched meta-variables.
3. *Patch*: replace the node text with the generated fix.

Let's see a simple example: replace `console.log` with `logger.log`. The following rule will do the trick.

```yaml
rule:
  pattern: console.log($MSG)
fix: logger.log($MSG)
```

The rule above is quite straightforward. It matches the `console.log` call, using the pattern, and replaces it with the `logger.log` call.
The meta-variable `$MSG` captures the argument of `console.log` and is used in the `fix` field.

ast-grep also has several other fields to fine-tune the process. The core fields in ast-grep's rule map naturally to the idea of **Find & Patch**.

* **Find**
  * Find a target node based on the [`rule`](https://ast-grep.github.io/reference/rule.html)
  * Filter the matched nodes based on [`constraints`](https://ast-grep.github.io/reference/yaml.html#constraints)
* **Patch**
  * Rewrite the matched meta-variable based on [`transform`](https://ast-grep.github.io/reference/yaml/transformation.html)
  * Replace the matched node with [`fix`](https://ast-grep.github.io/reference/yaml/fix.html), which can use the transformed meta-variables.

## Limitation of the Current Workflow

However, this workflow has a limitation: it can only replace one node at a time, which means that we cannot handle complex transformations that involve multiple nodes or lists of nodes.

For example, suppose we want to rewrite barrel imports to single imports. A [barrel import](https://adrianfaciu.dev/posts/barrel-files/) is a way to consolidate the exports of multiple modules into a single convenient module that can be imported using a single import statement. For instance:

```js
import {a, b, c} from './barrel';
```

This imports three modules (`a`, `b`, and `c`) from a single barrel file (`barrel.js`) that re-exports them.

Rewriting this to single imports has [some](https://vercel.com/blog/how-we-optimized-package-imports-in-next-js) [benefits](https://marvinh.dev/blog/speeding-up-javascript-ecosystem-part-7/), such as reducing [bundle size](https://dev.to/tassiofront/barrel-files-and-why-you-should-stop-using-them-now-bc4) or avoiding [conflicting names](https://flaming.codes/posts/barrel-files-in-javascript/).

```js
import a from './barrel/a';
import b from './barrel/b';
import c from './barrel/c';
```

This imports each module directly from its own file, without going through the barrel file.

With the simple "Find and Patch" workflow, we cannot achieve this transformation easily. We either have to rewrite the whole import statement or rewrite each identifier one by one. We cannot replace the whole import statement because we cannot process the multiple identifiers, which requires processing a list of nodes at one time.
Can we rewrite the identifiers one by one? This also fails because we cannot replace the whole import statement, so there will be unwanted import statement text surrounding the identifiers.

```javascript
// we cannot rewrite the whole import statements
// because we don't know how to rewrite a, b, c as a list
import ??? from './barrel';
// we cannot rewrite each identifier
// because the replaced text is inside the import statement
import { ??, ??, ?? } from './barrel';
```

We need a better way to rewrite code that involves multiple nodes or lists of nodes. And here comes **Find & Patch**.

## Extend the Concept of `Find` and `Patch`

Let's reflect: what limits us from rewriting the code above?

Our old workflow does not allow us to apply a rule to multiple sub-nodes of a node. (This is like not being able to write for loops.)

Nor does it allow us to generate different text for different sub-nodes in a rule. (This is like not being able to write if/switch statements.)

I initially thought of adding [list comprehension](https://github.com/ast-grep/ast-grep/issues/723#issuecomment-1890362116) to transform to overcome these limitations. However, list comprehension will introduce more concepts like loops, filters and probably nested loops. I prefer having [Occam's razor](https://www.wikiwand.com/en/Occam%27s_razor) to shave off unnecessary constructs.

Luckily, [Mosenkis](https://github.com/emosenkis) proposed the [refreshing idea](https://github.com/ast-grep/ast-grep/issues/723#issuecomment-1883526774) that we can apply sub-rules, called `rewriters`, to specific nodes during matching. It can elegantly solve the issue of processing multiple nodes with multiple different rules!

The idea is simple: we will add three new, but similar, steps in the rewriting step.

1. *Find* a list of different sub-nodes under a meta-variable that match different rewriters.
2. *Generate* a different fix for each sub-node based on the matched rewriter sub-rule.
3. *Join* the fixes together and store the string in a new metavariable for later use.

The new steps are similar to the existing **"Find and Patch"** workflow. It is like recursively applying the old workflow to matched nodes!

We can, taking the previous barrel import as an example, first match the import statement and then apply the rewriter sub-rule to each identifier.

## Intriguing Example

The idea above is implemented by a new [`rewriters`](https://ast-grep.github.io/reference/yaml/rewriter.html) field and a new [`rewrite`](https://ast-grep.github.io/reference/yaml/transformation.html#rewrite) transformation.

**Our first step is to write a rule to capture the import statement.**

```yaml
rule:
  pattern: import {$$$IDENTS} from './barrel'
```

This will capture the imported identifiers `a, b, c` in `$$$IDENTS`.

**Next, we need to transform `$$$IDENTS` to individual imports.**

The idea is that we can find the identifier nodes in the `$$$IDENT` and rewrite them to individual imports.

To do this, we register a rewriter that acts as a separate rewriter rule for each identifier.

```yaml
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  fix: import $IDENT from './barrel/$IDENT'
```

The `rewrite-identifier` above will:

1. First, find each `identifier` AST node and capture it as `$IDENT`.
2. Rewrite the identifier to a new import statement.

For example, the rewriter will change identifier `a` to  `import a from './barrel/a'`.

**We can now apply the rewriter to the matched variable `$$$IDENTS`.**

The counterpart of `rewriter` is the `rewrite` transformation, which applies the rewriter to a matched variable and generates a new string.

The yaml fragment below uses `rewrite` to find identifiers in `$$$IDENTS`, as specified in `rewrite-identifier`'s rule,
and rewrites it to single import statement.

```yaml
transform:
  IMPORTS:
    rewrite:
      rewriters: [rewrite-identifer]
      source: $$$IDENTS
      joinBy: "\n"
```

Note the `joinBy` field in the transform section. It specifies how to join the rewritten import statements with a newline character. This means that each identifier will generate a separate import statement, followed by a newline.

**Finally, we can use the transformed `IMPORTS` in the `fix` field to replace the original import statement.**

The final rule will be like this. See the [online playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBpbXBvcnQgeyQkJElERU5UU30gZnJvbSAnLi9iYXJyZWwnXG5yZXdyaXRlcnM6XG4tIGlkOiByZXdyaXRlLWlkZW50aWZlclxuICBydWxlOlxuICAgIHBhdHRlcm46ICRJREVOVFxuICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgZml4OiBpbXBvcnQgJElERU5UIGZyb20gJy4vYmFycmVsLyRJREVOVCdcbnRyYW5zZm9ybTpcbiAgSU1QT1JUUzpcbiAgICByZXdyaXRlOlxuICAgICAgcmV3cml0ZXJzOiBbcmV3cml0ZS1pZGVudGlmZXJdXG4gICAgICBzb3VyY2U6ICQkJElERU5UU1xuICAgICAgam9pbkJ5OiBcIlxcblwiXG5maXg6ICRJTVBPUlRTIiwic291cmNlIjoiaW1wb3J0IHsgYSwgYiwgYyB9IGZyb20gJy4vYmFycmVsJzsifQ==).

```yaml
rule:
  pattern: import {$$$IDENTS} from './barrel'
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  fix: import $IDENT from './barrel/$IDENT'
transform:
  IMPORTS:
    rewrite:
      rewriters: [rewrite-identifer]
      source: $$$IDENTS
      joinBy: "\n"
fix: $IMPORTS
```

## Similarity to Functional Programming

Find & Patch is a scheme that allows us to manipulate the syntax tree of the code in a declarative way.

It reminds me of Rust declarative macro since both Find & Patch and Rust declarative macro can:

* Match a list of nodes/tokens based on patterns: ast-grep's rule vs. Rust macro pattern matcher.
* Break nodes/tokens into sub parts: ast-grep's metavariable vs. Rust macro variable.
* Recursively use subparts to call other rewrite/macros.

The idea can be further compared to functional programming! We can use different rules to match and transform different sub-nodes of the tree, just like using [pattern matching](https://www.wikiwand.com/en/Pattern_matching) in functional languages. We can also apply rules to multiple sub-nodes at once, just like using for-comprehension or map/filter/reduce. Moreover, we can break down a large syntax tree into smaller sub-trees by using meta-variables, just like using destructuring or [elimination rules](https://blog.jez.io/intro-elim/) in functional languages. But all of these can be boiled down to two simple idea: **Finding** nodes and **Patching** nodes!

Find & Patch is a simple and elegant scheme that is tailored for AST manipulation, but it can achieve similar transformations as a general-purpose functional programming language doing rewrites!

We can think of Find & Patch as a form of "Functional Programming" over the AST! And they both have the same acronym btw.

***

Hope you find this scheme useful and interesting, and I sincerely invite you to try it out with ast-grep. Thank you for reading~

---

---
url: /advanced/custom-language.md
---
# Custom Language Support

In this guide, we will show you how to use a custom language that is not built into ast-grep.

We will use [Mojo 🔥](https://www.modular.com/mojo) as an example!

***

[Tree-sitter](https://tree-sitter.github.io/tree-sitter/) is a popular parser generator library that ast-grep uses to support many languages.
However, not all Tree-sitter compatible languages are shipped with ast-grep command line tool.

If you want to use a custom language that is not built into ast-grep, you can compile it as a dynamic library first and load it via custom language registration.

There will be three steps to achieve this:

1. Install tree-sitter CLI and prepare the grammar file.
2. Compile the custom language as a dynamic library.
3. Register the custom language in ast-grep project config.

:::tip Pro Tip
You can also reuse the dynamic library compiled by neovim. See [this link](https://github.com/nvim-treesitter/nvim-treesitter/#changing-the-parser-install-directory) to find where the parsers are.
:::

## Prepare Tree-sitter Tool and Parser

Before you can compile a custom language as a dynamic library, you need to install the Tree-sitter CLI tool and get the Tree-sitter grammar for your custom language.

The recommended way to install the Tree-sitter CLI tool is via [npm](https://www.npmjs.com/package/tree-sitter-cli):

```bash
npm install -g tree-sitter-cli
```

Alternative installation methods are also available in the [official doc](https://tree-sitter.github.io/tree-sitter/creating-parsers#installation).

For the Tree-sitter grammar, you can either [write your own](https://tree-sitter.github.io/tree-sitter/creating-parsers#writing-the-grammar) or find one from the Tree-sitter grammars [repository](https://github.com/tree-sitter).

Since **Mojo** is a new language, we cannot find an existing repo for it. But I have created a mock [grammar for Mojo](https://github.com/HerringtonDarkholme/tree-sitter-mojo).

You can clone it for the tutorial sake. It is forked from Python and barely contains Mojo syntax(just `struct`/`fn` keywords).

```bash
git clone https://github.com/HerringtonDarkholme/tree-sitter-mojo.git
```

## Compile the Parser as Dynamic Library

Once we have prepared the tool and the grammar, we can compile the parser as dynamic library.
*`tree-sitter-cli` is the preferred way to compile dynamic library.*

The [official way](https://tree-sitter.github.io/tree-sitter/cli/build.html) to compile a parser as a dynamic library is to use the `tree-sitter build` command.

```sh
tree-sitter build --output mojo.so
```

The build command compiles your parser into a dynamically-loadable library as a shared object (.so, .dylib, or .dll).

Another way is to use the following [commands](https://github.com/tree-sitter/tree-sitter/blob/a62bac5370dc5c76c75935834ef083457a6dd0e1/cli/loader/src/lib.rs#L380-L410) to compile the parser manually:

```shell
gcc -shared -fPIC -fno-exceptions -g -I {header_path} -o {lib_path} -O2 {scanner_path} -xc {parser_path} {other_flags}
```

where `{header_path}` is the path to the folder of header file of your custom language parser (usually `src`) and `{lib_path}` is the path where you want to store the dynamic library (in this case `mojo.so`). `{scanner_path}` and `{parser_path}` are the `c` or `cc` files of your parser. You also need to include other gcc flags if needed.

For example, in mojo's case, the full command will be:

```shell
gcc -shared -fPIC -fno-exceptions -g -I 'src' -o mojo.so -O2 src/scanner.cc -xc src/parser.c -lstdc++
```

:::details Old tree-sitter does not have build command

[Previously](https://github.com/tree-sitter/tree-sitter/pull/3174) there are no official instructions on how to do this on the internet, but we can get some hints from Tree-sitter's [source code](https://github.com/tree-sitter/tree-sitter/blob/a62bac5370dc5c76c75935834ef083457a6dd0e1/cli/loader/src/lib.rs#L111).

One way is to set an environment variable called `TREE_SITTER_LIBDIR` to the path where you want to store the dynamic library, and then run `tree-sitter test` in the directory of your custom language parser.

This will generate a dynamic library at the `TREE_SITTER_LIBDIR` path.

For example:

```sh
cd path/to/mojo/parser
export TREE_SITTER_LIBDIR=path/to/your/dir
tree-sitter test
```

:::

## Register Language in `sgconfig.yml`

Once you have compiled the dynamic library for your custom language, you need to register it in the `sgconfig.yml` file.
You can use the command [`ast-grep new`](/guide/scan-project.html#create-scaffolding) to create a project and find the configuration file in the project root.

You need to add a new entry under the `customLanguages` key with the name of your custom language and some properties:

```yaml
# sgconfig.yml
ruleDirs: ["./rules"]
customLanguages:
  mojo:
      libraryPath: mojo.so     # path to dynamic library
      extensions: [mojo, 🔥]   # file extensions for this language
      expandoChar: _           # optional char to replace $ in your pattern
```

The `libraryPath` property specifies the path to the dynamic library relative to the `sgconfig.yml` file or an absolute path. The `extensions` property specifies a list of file extensions for this language.
The `expandoChar` property is optional and specifies a character that can be used instead of `$` for meta-variables in your pattern.

:::tip What's expandoChar?
ast-grep requires pattern to be a valid syntactical construct, but `$VAR` might not be a valid expression in some language.
`expandoChar` will replace `$` in the pattern so it can be parsed successfully by Tree-sitter.
:::

For example, `$VAR` is not valid in ~~[Python](https://github.com/ast-grep/ast-grep/blob/1b999b249110c157ae5026e546a3112cd64344f7/crates/language/src/python.rs#L15)~~ Mojo. So we need to replace it with `_VAR`.
You can check the `expandoChar` of ast-grep's built-in languages [here](https://github.com/ast-grep/ast-grep/tree/main/crates/language/src).

## Use It!

Now you are ready to use your custom language with ast-grep! You can use it as any other supported language with the `-l` flag or the `language` property in your rule.

For example, to search for all occurrences of `print` in mojo files, you can run:

```bash
ast-grep -p "print" -l mojo
```

Or you can write a rule in yaml like this:

```yaml
id: my-first-mojo-rule
language: mojo  # the name we register before!
severity: hint
rule:
  pattern: print
```

And that's it! You have successfully used a custom language with ast-grep!

## Inspect Parser Output

Due to limited bandwidth, ast-grep does not support pretty print Concrete Syntax Trees.

However, you can use [tree-sitter-cli](https://github.com/tree-sitter/tree-sitter/tree/master/cli#commands) to dump the AST tree for your file.

```bash
tree-sitter parse [file_path]
```

:::warning Quiz Time
Can you support parse `main.ʕ◔ϖ◔ʔ` as [Golang](https://github.com/golang/go/issues/59968)?

[Answer](https://twitter.com/hd_nvim/status/1655085184855969797).
:::

---

---
url: /advanced/language-injection.md
---
# Search Multi-language Documents in ast-grep

## Introduction

ast-grep works well searching files of one single language, but it is hard to extract a sub language embedded inside a document.

However, in modern development, it's common to encounter **multi-language documents**. These are source files containing code written in multiple different languages. Notable examples include:

* **HTML files**: These can contain JavaScript inside `<script>` tags and CSS inside `<style>` tags.
* **JavaScript files**: These often contain regular expression, CSS style and query languages like graphql.
* **Ruby files**: These can contain snippets of code inside heredoc literals, where the heredoc delimiter often indicates the language.

These multi-language documents can be modeled in terms of a parent syntax tree with one or more *injected syntax trees* residing *inside* certain nodes of the parent tree.

ast-grep now supports a feature to handle **language injection**, allowing you to search for code written in one language within documents of another language.

This concept and terminology come from [tree-sitter's language injection](https://tree-sitter.github.io/tree-sitter/syntax-highlighting#language-injection), which implies you can *inject* another language into a language document. (BTW, [neovim](https://github.com/nvim-treesitter/nvim-treesitter?tab=readme-ov-file#adding-queries) also embraces this terminology.)

## Example: Search JS/CSS in the CLI

Let's start with a simple example of searching for JavaScript and CSS within HTML files using ast-grep's command-line interface (CLI).
ast-grep has builtin support to search JavaScript and CSS inside HTML files.

### **Using `ast-grep run`**: find patterns of CSS in an HTML file

Suppose we have an HTML file like below:

```html
<style>
  h1 { color: red; }
</style>
<h1>
  Hello World!
</h1>
<script>
  alert('hello world!')
</script>
```

Running this ast-grep command will extract the matching CSS style code out of the HTML file!

```sh
ast-grep run -p 'color: $COLOR'
```

ast-grep outputs this beautiful CLI report.

```shell
test.html
2│  h1 { color: red; }
```

ast-grep works well even if just providing the pattern without specifying the pattern language!

### **Using `ast-grep scan`**: find JavaScript in HTML with rule files

You can also use ast-grep's [rule file](https://ast-grep.github.io/guide/rule-config.html) to search injected languages.

For example, we can warn the use of `alert` in JavaScript, even if it is inside the HTML file.

```yml
id: no-alert
language: JavaScript
severity: warning
rule:
  pattern: alert($MSG)
message: Prefer use appropriate custom UI instead of obtrusive alert call.
```

The rule above will detect usage of `alert` in JavaScript. Running the rule via `ast-grep scan`.

```sh
ast-grep scan --rule no-alert.yml
```

The command leverages built-in behaviors in ast-grep to handle language injection seamlessly. It will produce the following warning message for the HTML file above.

```sh
warning[no-alert]: Prefer use appropriate custom UI instead of obtrusive alert call.
  ┌─ test.html:8:3
  │
8 │   alert('hello world!')
  │   ^^^^^^^^^^^^^^^^^^^^^
```

## How language injections work?

ast-grep employs a multi-step process to handle language injections effectively. Here's a detailed breakdown of the workflow:

1. **File Discovery**: The CLI first discovers files on the disk via the venerable [ignore](https://crates.io/crates/ignore) crate, the same library under [ripgrep](https://github.com/BurntSushi/ripgrep)'s hood.

2. **Language Inference**: ast-grep infers the language of each discovered file based on file extensions.

3. **Injection Extraction**: For documents that contain code written in multiple languages (e.g., HTML with embedded JS), ast-grep extracts the injected language sub-regions. *At the moment, ast-grep handles HTML/JS/CSS natively*.

4. **Code Matching**: ast-grep matches the specified patterns or rules against these regions. Pattern code will be interpreted according to the injected language (e.g. JS/CSS), instead of the parent document language (e.g. HTML).

## Customize Language Injection: styled-components in JavaScript

You can customize language injection via the `sgconfig.yml` [configuration file](https://ast-grep.github.io/reference/sgconfig.html). This allows you to specify how ast-grep handles multi-language documents based on your specific needs, without modifying ast-grep's built-in behaviors.

Let's see an example of searching CSS code in JavaScript. [styled-components](https://styled-components.com/) is a library for styling React applications using [CSS-in-JS](https://bootcamp.uxdesign.cc/css-in-js-libraries-for-styling-react-components-a-comprehensive-comparison-56600605a5a1). It allows you to write CSS directly within your JavaScript via [tagged template literals](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Template_literals), creating styled elements as React components.

The example will configure ast-grep to detect styled-components' CSS.

### Injection Configuration

You can add the `languageInjections` section in the project configuration file `sgconfig.yml`.

```yaml
languageInjections:
- hostLanguage: js
  rule:
    pattern: styled.$TAG`$CONTENT`
  injected: css
```

Let's break the configuration down.

1. `hostLanguage`: Specifies the main language of the document. In this example, it is set to `js` (JavaScript).

2. `rule`: Defines the ast-grep rule to identify the injected language region within the host language.

   * `pattern`: The pattern matches styled components syntax where `styled` is followed by a tag (e.g., `button`, `div`) and a template literal containing CSS.
   * the rule should have a meta variable `$CONTENT` to specify the subregion of injected language. In this case, it is the content inside the template string.

3. `injected`: Specifies the injected language within the identified regions. In this case, it is `css`.

### Example Match

Consider a JSX file using styled components:

```js
import styled from 'styled-components';

const Button = styled.button`
  background: red;
  color: white;
  padding: 10px 20px;
  border-radius: 3px;
`

export default function App() {
  return <Button>Click Me</Button>
}
```

With the above `languageInjections` configuration, ast-grep will:

1. Identify the `styled.button` block as a CSS region.
2. Extract the CSS code inside the template literal.
3. Apply any CSS-specific pattern searches within this extracted region.

You can search the CSS inside JavaScript in the project configuration folder using this command:

```sh
ast-grep -p 'background: $COLOR' -C 2
```

It will produce the match result:

```shell
styled.js
2│
3│const Button = styled.button`
4│  background: red;
5│  color: white;
6│  padding: 10px 20px;
```

## Using Custom Language with Injection

Finally, let's look at an example of searching for GraphQL within JavaScript files.
This demonstrates ast-grep's flexibility in handling custom language injections.

### Define graphql custom language in `sgconfig.yml`.

First, we need to register graphql as a custom language in ast-grep. See [custom language reference](https://ast-grep.github.io/advanced/custom-language.html) for more details.

```yaml
customLanguages:
  graphql:
    libraryPath: graphql.so # the graphql tree-sitter parser dynamic library
    extensions: [graphql]   # graphql file extension
    expandoChar: $          # see reference above for explanation
```

### Define graphql injection in `sgconfig.yml`.

Next, we need to customize what region should be parsed as graphql string in JavaScript. This is similar to styled-components example above.

```yaml
languageInjections:
- hostLanguage: js
  rule:
    pattern: graphql`$CONTENT`
  injected: graphql
```

### Search GraphQL in JavaScript

Suppose we have this JavaScript file from [Relay](https://relay.dev/), a GraphQL client framework.

```js
import React from "react"
import { graphql } from "react-relay"

const artistsQuery = graphql`
  query ArtistQuery($artistID: String!) {
    artist(id: $artistID) {
      name
      ...ArtistDescription_artist
    }
  }
`
```

We can search the GraphQL fragment via this `--inline-rules` scan.

```sh
ast-grep scan --inline-rules="{id: test, language: graphql, rule: {kind: fragment_spread}}"
```

Output

```sh
help[test]:
  ┌─ relay.js:8:7
  │
8 │       ...ArtistDescription_artist
  │       ^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

## More Possibility to be Unlocked...

By following these steps, you can effectively use ast-grep to search and analyze code across multiple languages within the same document, enhancing your ability to manage and understand complex codebases.

This feature extends to various frameworks like [Vue](https://vuejs.org/) and [Svelte](https://svelte.dev/), enables searching for [SQL in React server actions](https://x.com/peer_rich/status/1717609270475194466), and supports new patterns like [Vue-Vine](https://x.com/hd_nvim/status/1815300932793663658).

Hope you enjoy the feature! Happy ast-grepping!

---

---
url: /advanced/tool-comparison.md
---
# Comparison With Other Frameworks

:::danger Disclaimer
This comparison is based on the author's personal experience and opinion, which may not be accurate or comprehensive.
The author respects and appreciates all the other tools and their developers, and does not intend to criticize or endorse any of them.
The author is grateful to these predecessor tools for inspiring ast-grep! The reader is encouraged to try out the tools themselves and form their own judgment.
:::

## ast-grep

**Pros**:

* It is very performant. It uses [ignore](https://docs.rs/ignore/latest/ignore/) to do multi-thread processing, which makes it utilize all your CPU cores.
* It is language aware. It uses tree-sitter, a real parser, to parse the code into ASTs, which enables more precise and accurate matching and fixing.
* It has a powerful and flexible rule system. It allows you to write patterns, AST types and regular expressions to match code. It provides operators to compose complex matching rules for various scenarios.
* It can be used as a lightweight CLI tool or as a library, depending on your usage. It has a simple and user-friendly interface, and it also exposes its core functionality as a library for other applications.

**Cons**:

* It is still young and under development. It may have some bugs or limitations that need to be fixed or improved.
* It does not have deep semantic information or comparison equivalence. It only operates on the syntactic level of the code, which may miss some matches or may be too cumbersome to match certain code.
* More specifically, ast-grep at the moment does not support the following information:
  * [type information](https://semgrep.dev/docs/writing-rules/pattern-syntax#typed-metavariables)
  * [control flow analysis](https://en.wikipedia.org/wiki/Control-flow_analysis)
  * [data flow analysis](https://en.wikipedia.org/wiki/Data-flow_analysis)
  * [taint analysis](https://semgrep.dev/docs/writing-rules/data-flow/taint-mode)
  * [constant propagation](https://semgrep.dev/docs/writing-rules/data-flow/constant-propagation)

## [Semgrep](https://semgrep.dev/)

Semgrep is a well-established tool that uses code patterns to find and fix bugs and security issues in code.

**Pros**:

* It supports advanced features like equivalence and deep-semgrep, which allow for more precise and expressive matching and fixing.
* It has a large collection of rules for various languages and frameworks, which cover common vulnerabilities and best practices.

**Cons**:

* It is mainly focused on security issues, which may limit its applicability for other use cases.
* It is relatively slow when used as command line tools.
* It cannot be used as a library in other applications, which may reduce its integration and customization options.

## [GritQL](https://about.grit.io/)

[GritQL](https://docs.grit.io/language/overview) language is [Grit](https://docs.grit.io/)'s embedded query language for searching and transforming source code.

**Pros**:

* GritQL is generally more powerful. It has features like [clause](https://docs.grit.io/language/modifiers) from [logic programming language](https://en.wikipedia.org/wiki/Logic_programming#:~:text=A%20logic%20program%20is%20a,Programming%20\(ASP\)%20and%20Datalog.) and [operations](https://docs.grit.io/language/conditions#match-condition) from imperative programming languages.
* It is used as [linter plugins](https://biomejs.dev/linter/plugins/) in [Biome](https://biomejs.dev/), a toolchain for JS ecosystem.

**Cons**:

* Depending on different background, developers may find it harder to learn a multi-paradigm DSL.

## [Comby](https://comby.dev/)

Comby is a fast and flexible tool that uses structural patterns to match and rewrite code across languages and file formats.

**Pros**:

* It does not rely on language-specific parsers, which makes it more generic and robust. It can handle any language and file format, including non-code files like JSON or Markdown.
* It has a custom syntax for specifying patterns and replacements, which can handle various syntactic variations and transformations.

**Cons**:

* It is not aware of the syntax and semantics of the target language, which limits its expressiveness and accuracy. It may miss some matches or generate invalid code due to syntactic or semantic differences.
* It does not support indentation-sensitive languages like Python or Haskell, which require special handling for whitespace and indentation.
* It is hard to write complex queries with Comby, such as finding a function that does not call another function. It does not support logical operators or filters for patterns.

## [IntelliJ Structural Search Replace](https://www.jetbrains.com/help/idea/structural-search-and-replace.html)

IntelliJ Structural Search Replace is not a standalone tool, but a feature of the IntelliJ IDE that allows users to search and replace code using structural patterns.

**Pros**:

* It is integrated with the IntelliJ IDE, which makes it easy to use and customize.

**Cons**:

* Currently, IntelliJ IDEA supports the structural search and replace for Java, Kotlin and Groovy.

---

---
url: /contributing/how-to.md
---
# Contributing

:tada: ***We are thrilled that you are interested in contributing to the ast-grep project!*** :tada:

Your help and support are very valuable for us.
There are many ways you can help improve the project and make it more useful for everyone.

Let's see some of the things we can do together:

## Spreading Your Words ❤️

We appreciate your kind words and support for the project. You can help us grow the ast-grep community and reach more potential users by spreading your kind words. Here are some of the things we can do:

* **Who is using ast-grep**: Let us know who is using ast-grep by adding your name or organization to the [users page](https://github.com/ast-grep/ast-grep/issues/373) on the documentation website. Feel free to add a logo or a testimonial if you like.

* **Tweet it!**: Tweet about ast-grep using the hashtag [#ast\_grep](https://twitter.com/hashtag/ast_grep). Share your feedback, your use cases, your tips and tricks, or your questions and suggestions with the ast-grep community on Twitter.

* **Sharing Podcast**: Talk about ast-grep on podcasts or other audio platforms. Introduce ast-grep to new audiences, share your stories and insights, or invite other guests to discuss ast-grep with you.

* **Meetup**: Attend meetups or events where you can talk about ast-grep. Meet other ast-grep users or developers, exchange ideas and experiences, learn from each other, or collaborate on projects.

## Giving Feedback

We appreciate your feedback on the project. Whether you have a feature request, a bug report, or a general comment, we would love to hear from you. You can use the following channels to provide your feedback:

* **Feature Request**: If you have an idea for a new feature or an enhancement for an existing feature, please create an issue on the [main repo](https://github.com/ast-grep/ast-grep/issues/new?assignees=\&labels=enhancement\&projects=\&template=feature_request.md\&title=%5Bfeature%5D) with the label `enhancement`. Please describe your idea with examples and explain why it would be useful for the project and the users.

* **Bug Report**: If you encounter a bug or an error while using ast-grep, please create an issue on the [main repo](https://github.com/ast-grep/ast-grep/issues/new?assignees=\&labels=enhancement\&projects=\&template=feature_request.md\&title=%5Bfeature%5D) with the label `bug`. Please provide as much information as possible to help us reproduce and fix the bug, such as the version of ast-grep, the command or query you used, the expected and actual results, any error messages or screenshots, and preferably a [playground link](/playground.html) reproducing the issue.

## Contributing Code

We welcome your code contributions to the project. Whether you want to fix a bug, implement a feature, improve the documentation, or add a new integration, we are grateful for your help. You can use the following repositories to contribute your code:

* **CLI Main Repo**: The [main repository for ast-grep](https://github.com/ast-grep/ast-grep) command-line interface (CLI). It contains the core logic and functionality of ast-grep. For small features or typo fixes, you can fork this repository and submit pull requests with your changes. [This guide](/contributing/development.html) may help you set up essential tools for development. *For larger features or big changes, please make an issue for discussion before jumping into it.*

- **Doc Website**: This is the repository for the ast-grep documentation website. It contains the source files for generating the website using [vitepress](https://vitepress.dev/). You can fork this repository and submit pull requests with your changes.&#x20;

- **CI/CD Integration**: ast-grep has a [repository for GitHub Action](https://github.com/ast-grep/action). It allows you to use ast-grep as part of your continuous integration and continuous delivery (CI/CD) workflows on GitHub. You can check this repository and suggest useful features missing now.

- **Editor Integration**: These are the repositories for various editor integrations of ast-grep. They allow you to use ast-grep within your favorite editor, such as VS Code, Vim, or Neovim. Please follow the respective guides for each editor integration before submitting your pull requests.
  * VS Code extension: [ast-grep-vscode](https://github.com/ast-grep/ast-grep-vscode)
  * NeoVim LSP: [coc-ast-grep](https://github.com/yaegassy/coc-ast-grep) made by [@yaegassy](https://twitter.com/yaegassy)
  * NeoVim Telescope plugin: [telescope-sg](https://github.com/Marskey/telescope-sg) made by [@Marskey](https://github.com/Marskey)

## Sharing Knowledge

We encourage you to share your knowledge and experience with ast-grep with others. You can help us spread the word about ast-grep and educate more people about its benefits and features. Here are some of the things we can do:

* **Write introductions to ast-grep**: You can write blog posts, articles, or tutorials that introduce ast-grep to new users. You can explain what ast-grep is, how it works, what problems it solves, and how to install and use it. You can also share some examples of how you use ast-grep in your own projects or workflows.

* **Answer questions about ast-grep**: Help answering people's questions on [StackOverflow](https://stackoverflow.com/questions/tagged/ast-grep) or [Discord](https://discord.gg/4YZjf6htSQ). Your answers will be appreciated!

* **Write ast-grep's tutorial**: You can write more advanced tutorials that show how to use ast-grep for specific tasks or scenarios. You can demonstrate how to use ast-grep's features and options, how to write complex queries and transformations, how to integrate ast-grep with other tools or platforms, and how to optimize ast-grep's performance and efficiency.

* **Translate documentation**: You can help us make ast-grep more accessible to users from different regions and languages by translating its documentation into other languages. Reach out [@Shenqingchuan](https://twitter.com/Shenqingchuan), translation team member of [Rollup](https://github.com/rollup/rollup-docs-cn), [Vite](https://github.com/vitejs/docs-cn) and ast-grep, for more ideas about translation!

- **Curate a rule collections**: Using ast-grep as linter in your project can showcase the power and versatility of ast-grep! Linting open source projects shows how ast-grep can be used for various purposes and domains. [ast-grep/eslint](https://github.com/ast-grep/eslint), for example, is a collection of eslint rule implemented in ast-grep YAML.

- **Sharing Rules**: Sharing your rules on ast-grep's [example catalog](/catalog/index.html) can inspire more people to harness the power of AST! Example catalog is a place where users can browse, search, and submit rules. You can use [the template](https://github.com/ast-grep/ast-grep.github.io/blob/main/website/catalog/rule-template.md) to add your example [here](https://github.com/ast-grep/ast-grep.github.io/tree/main/website/catalog).

Thank you for your interest in contributing to the ast-grep project. We are grateful for your help and support. We hope you enjoy using and improving ast-grep as much as we do. If you have any questions or issues, please feel free to contact us on [GitHub](https://github.com/ast-grep/ast-grep) or [Discord](https://discord.gg/4YZjf6htSQ). We look forward to hearing from you soon! 😊

***

:::tip You don’t have to contribute code

A common misconception about contributing to open source is that you need to contribute code. In fact, it’s often the other parts of a project that are most neglected or overlooked. You’ll do the project a huge favor by offering to pitch in with these types of contributions!

*[GitHub Open Source Guide](https://opensource.guide/)*

---

---
url: /contributing/development.md
---
# Development Guide

## Environment Setup

ast-grep is written in [Rust](https://www.rust-lang.org/) and hosted by [git](https://git-scm.com/).

You need to have rust environment installed to build ast-grep.
The recommended way to install rust is via [rustup](https://rustup.rs/).
Once you have rustup installed, you can install rust by running:

```bash
rustup install stable
```

You also need  [prek](https://github.com/j178/prek) to setup git hooks for type checking, formatting and clippy.

Run prek install to set up the git hook scripts.

```bash
prek install
```

Optionally, you can also install [nodejs](https://github.com/Schniz/fnm) and [yarn](https://yarnpkg.com/) for napi binding development.

That's it! You have setup the environment for ast-grep!

## Common Commands

The below are some cargo commands common to any Rust project.

```bash
cargo test     # Run test
cargo check    # Run checking
cargo clippy   # Run clippy
cargo fmt      # Run formatting
```

Below are some ast-grep specific commands.

## N-API Development

[@ast-grep/napi](https://www.npmjs.com/package/@ast-grep/napi) is the [nodejs binding](https://napi.rs/) for ast-grep.

The source code of napi binding is under the `crates/napi` folder. You can refer to the [package.json](https://github.com/ast-grep/ast-grep/blob/main/crates/napi/package.json) for available commands.

```bash
cd crates/napi
yarn   # Install dependencies
yarn build # Build the binding
yarn test # Run test
```

## Commit Conventions

ast-grep loosely follows the [commit conventions](https://www.conventionalcommits.org/en/v1.0.0/).

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

To quote the conventional commits doc:

> The commit contains the following structural elements, to communicate intent to the consumers of your library:
>
> * `fix:` a commit of the type fix patches a bug in your codebase.
> * `feat:` a commit of the type feat introduces a new feature to the codebase.
> * types other than `fix:` and `feat:` are allowed, for example, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, and `test:`.
> * `BREAKING CHANGE`: a commit that has a footer `BREAKING CHANGE:` introduces a breaking API change. A `BREAKING CHANGE` can be part of commits of any type.
> * footers other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to git trailer format.

:::tip
`BREAKING CHANGE` will be picked up and written in `CHANGELOG` by [`cargo xtask`](https://github.com/ast-grep/ast-grep/blob/86afc5865b42285106f232f01c0eb45708d134c3/xtask/src/main.rs#L162-L171).
:::

## Run Benchmark

ast-grep's Benchmark is not included in the default cargo test. You need to run the benchmark command in `benches` folder.

```bash
cd benches
cargo bench
```

ast-grep's benchmarking suite is not well developed yet. The result may fluctuate too much.

## Release New Version

The command below will bump version and create a git tag for ast-grep.
Once pushed to GitHub, the tag will trigger [GitHub actions](https://github.com/ast-grep/ast-grep/blob/main/.github/workflows/coverage.yml) to build and publish the new version to [crates.io](https://github.com/ast-grep/ast-grep/blob/main/.github/workflows/pypi.yml), [npm](https://github.com/ast-grep/ast-grep/blob/main/.github/workflows/napi.yml) and [PyPi](https://github.com/ast-grep/ast-grep/blob/main/.github/workflows/pypi.yml).

```bash
cargo xtask [version-number]
```

See [xtask](https://github.com/ast-grep/ast-grep/blob/main/xtask/src/main.rs) file for more details.

---

---
url: /contributing/add-lang.md
---
# Add New Language to ast-grep

Thank you for your interest in adding a new language to ast-grep!
We appreciate your contribution to this project. Adding new languages will make the tool more useful and accessible to a wider range of users.

However, there are some requirements and constraints that you need to consider before you start. This guide will help you understand the process and the standards of adding a new language to ast-grep.

## Requirements and Constraints

To keep ast-grep lightweight and fast, we have several factors to consider when adding a new language.
As a rule of thumb, we want to limit the binary size of ast-grep under 10MB after zip compression.

* **Popularity of the language**. While the popularity of a language does not necessarily reflect its merits, our limited size budget allows us to only support languages that are widely used and have a large user base. Online sources like [TIOBE index](https://www.tiobe.com/tiobe-index/) or [GitHub Octoverse](https://octoverse.github.com/2022/top-programming-languages) can help one to check the popularity of the language.

- **Quality of the Tree-sitter grammar**.  ast-grep relies on [Tree-sitter](https://tree-sitter.github.io/tree-sitter/), a parser generator tool and a parsing library, to support different languages. The Tree-sitter grammar for the new language should be *well-written*, *up-to-date*, and *regularly maintained*. You can search [Tree-sitter on GitHub](https://github.com/search?q=tree-sitter\&type=repositories) or on [crates.io](https://crates.io/search?q=tree%20sitter).

- **Size of the grammar**. The new language's grammar should not be too complicated. Otherwise it may take too much space from other languages. You can also check the current size of ast-grep in the [releases page](https://github.com/ast-grep/ast-grep/releases).

- **Availability of the grammar on crates.io**. To ease the maintenance burden, we prefer to use grammars that are published on crates.io, Rust's package registry. If your grammar is not on crates.io, you need to publish it yourself or ask the author to do so.

***

Don't worry if your language is not supported by ast-grep. You can try ast-grep's [custom language support](/advanced/custom-language.html) and register your own Tree-sitter parser!

If your language satisfies the requirements above, congratulations! Let's see how to add it to ast-grep.

## Add to ast-grep Core

ast-grep has several distinct use cases: [CLI tool](https://crates.io/crates/ast-grep), [n-api lib](https://www.npmjs.com/package/@ast-grep/napi) and [web playground](https://ast-grep.github.io/playground.html).

Adding a language includes two steps. The first step is to add the language to ast-grep core.
The core repository is multi-crate workspace hosted at [GitHub](https://github.com/ast-grep/ast-grep). The relevant crate is [language](https://github.com/ast-grep/ast-grep/tree/main/crates/language), which defines the supported languages and their tree-sitter grammars.

We will use Ruby as an example to show how to add a new language to ast-grep core. You can see [the commit](https://github.com/ast-grep/ast-grep/commit/ffe14ceb8773c5d2b85559ff7455070e2a1a9388#diff-3590708789e9cdf7fa0421ecba544a69e9bbe8dd0915f0d9ff8344a9c899adfd) as a reference.

### Add Dependencies

1. Add `tree-sitter-[lang]` crate as `dependencies` to the [Cargo.toml](https://github.com/ast-grep/ast-grep/blob/main/crates/language/Cargo.toml#L13) in the `language` crate.

```toml
# Cargo.toml
[dependencies]
...
tree-sitter-ruby = {version = "0.20.0", optional = true } // [!code ++]
...
```

*Note the  `optional` attribute is required here.*

2. Add the `tree-sitter-[lang]` dependency in [`builtin-parser`](https://github.com/ast-grep/ast-grep/blob/e494500fc5d6994c20fe0102aa4b93d2108827bb/crates/language/Cargo.toml#L40) list.

```toml
# Cargo.toml
[features]
builtin-parser = [
  ...
  "tree-sitter-ruby",  // [!code ++]
  ...
]
```

The `builtin-parser` feature is used for command line tool. Web playground is not using the builtin parser so the dependency must be optional.

### Implement Parser

3. Add the parser function in [parsers.rs](https://github.com/ast-grep/ast-grep/blob/main/crates/language/src/parsers.rs), where tree-sitter grammars are imported.

```rust
#[cfg(feature = "builtin-parser")]
mod parser_implementation  {
  ...
  pub fn language_ruby() -> TSLanguage { // [!code ++]
    tree_sitter_ruby::language().into()  // [!code ++]
  }                                      // [!code ++]
  ...
}

#[cfg(not(feature = "builtin-parser"))]
mod parser_implementation  {
  impl_parsers!(
    ...
    language_ruby, // [!code ++]
    ...
  );
}
```

Note there are two places to add, one for `#[cfg(feature = "builtin-parser")]` and the other for `#[cfg(not(feature = "builtin-parser"))]`.

4. Implement `language` trait by using macro in [lib.rs](https://github.com/ast-grep/ast-grep/commit/ffe14ceb8773c5d2b85559ff7455070e2a1a9388#diff-1f2939360f8f95434ed23b53406eac0aa8b2f404171b63c6466bbdfda728c82d)

```rust
// lib.rs
impl_lang_expando!(Ruby, language_ruby, 'µ'); // [!code ++]
```

There are two macros, `impl_lang_expando` or `impl_lang`, to generate necessary methods required by ast-grep [`Language`](https://github.com/ast-grep/ast-grep/blob/e494500fc5d6994c20fe0102aa4b93d2108827bb/crates/core/src/language.rs#L12) trait.

You need to choose one of them to use for the new language. If the language does not allow `$` as valid identifier character and you need to customize the expando\_char, use `impl_lang_expando`.

You can reference the comment [here](https://github.com/ast-grep/ast-grep/blob/e494500fc5d6994c20fe0102aa4b93d2108827bb/crates/language/src/lib.rs#L1-L8) for more information.

### Register the New Language

6. Add new lang in [`SupportLang`](https://github.com/ast-grep/ast-grep/blob/e494500fc5d6994c20fe0102aa4b93d2108827bb/crates/language/src/lib.rs#L119) enum.

```rust
// lib.rs
pub enum SupportLang {
  ...
  Ruby, // [!code ++]
  ...
}
```

7. Add new lang in [`execute_lang_method`](https://github.com/ast-grep/ast-grep/blob/e494500fc5d6994c20fe0102aa4b93d2108827bb/crates/language/src/lib.rs#L229C14-L229C33)

```rust
// lib.rs
macro_rules! execute_lang_method {
  ($me: path, $method: ident, $($pname:tt),*) => {
    use SupportLang as S;
    match $me {
      ...
      S::Ruby => Ruby.$method($($pname,)*), // [!code ++]
    }
  }
}
```

7. Add new lang in [`all_langs`](https://github.com/ast-grep/ast-grep/blob/be10ff97d6d5adad4b524961d82e40ca76ab4259/crates/language/src/lib.rs#L143), [`alias`](https://github.com/ast-grep/ast-grep/blob/be10ff97d6d5adad4b524961d82e40ca76ab4259/crates/language/src/lib.rs#L188), [`extension`](https://github.com/ast-grep/ast-grep/blob/be10ff97d6d5adad4b524961d82e40ca76ab4259/crates/language/src/lib.rs#L281) and [`file_types`](https://github.com/ast-grep/ast-grep/blob/be10ff97d6d5adad4b524961d82e40ca76ab4259/crates/language/src/lib.rs#L331)

See this [commit](https://github.com/ast-grep/ast-grep/commit/ffe14ceb8773c5d2b85559ff7455070e2a1a9388#diff-1f2939360f8f95434ed23b53406eac0aa8b2f404171b63c6466bbdfda728c82d) for the detailed code change.

:::tip Find existing languages as reference
The rule of thumb to add a new language is to find a reference language that is already included in the language crate.
Then add your new language by searching and following the existing language.
:::

## Add to ast-grep Playground

Adding new language to web playground is a little bit more complex.

The playground has a standalone [repository](https://github.com/ast-grep/ast-grep.github.io) and we need to change code there.

### Prepare WASM

1. Set up Tree-sitter

First, we need to set up Tree-sitter development tools like. You can refer to the Tree-sitter setup section in this [link](/advanced/custom-language.html#prepare-tree-sitter-tool-and-parser).

2. Build WASM file

Then, in your parser repository, use this command to build a WASM file.

```bash
tree-sitter generate # if grammar is not generated before
tree-sitter build --wasm
```

Note you may need to install [docker](https://www.docker.com/) when building WASM files.

3. Move WASM file to the website [`public`](https://github.com/ast-grep/ast-grep.github.io/tree/main/website/public) folder.

You can also see other languages' WASM files in the public directory.
The file name is in the format of `tree-sitter-[lang].wasm`. The name will be used later in [`parserPaths`](https://github.com/ast-grep/ast-grep.github.io/blob/a2dce64dda67e1c0842b757fc692ffe05639e407/website/src/components/lang.ts#L4).

### Add language in Rust

You need to add the language in the [wasm\_lang.rs](https://github.com/ast-grep/ast-grep.github.io/blob/main/src/wasm_lang.rs).
More specifically, you need to add a new enum variant in [`WasmLang`](https://github.com/ast-grep/ast-grep.github.io/blob/a2dce64dda67e1c0842b757fc692ffe05639e407/src/wasm_lang.rs#L16), handle the new variant in [`execute_lang_method`](https://github.com/ast-grep/ast-grep.github.io/blob/a2dce64dda67e1c0842b757fc692ffe05639e407/src/wasm_lang.rs#L111) and implement [`FromStr`](https://github.com/ast-grep/ast-grep.github.io/blob/a2dce64dda67e1c0842b757fc692ffe05639e407/src/wasm_lang.rs#L48).

```rust
// new variant
pub enum WasmLang {
  // ...
  Swift, // [!code ++]
}

// handle variant in macro
macro_rules! execute_lang_method {
  ($me: path, $method: ident, $($pname:tt),*) => {
    use WasmLang as W;
    match $me {
      W::Swift => L::Swift.$method($($pname,)*), // [!code ++]
    }
  }
}

// impl FromStr
impl FromStr for WasmLang {
  // ...
  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "swift" => Swift, // [!code ++]
    })
  }
}
```

### Add language in TypeScript

Finally you need to add the language in TypeScript to make it available in playground.
The file is [lang.ts](https://github.com/ast-grep/ast-grep.github.io/blob/main/website/src/components/lang.ts). There are two changes need to make.

```typescript
// Add language parserPaths
const parserPaths = {
  // ...
  swift: 'tree-sitter-swift.wasm', // [!code ++]
}

// Add language display name
export const languageDisplayNames: Record<SupportedLang, string> = {
  // ...
  swift: 'Swift',
}
```

You can see Swift's support as the [reference commit](https://github.com/ast-grep/ast-grep.github.io/commit/55a546535dee989ce5ee2582080e771d006d165e).

---

---
url: /playground.md
description: >-
  ast-grep playground is an online tool that lets you explore AST, debug custom
  lint rules, and inspect code rewriting with instant feedback.
---


---

---
url: /catalog/python/remove-async-await.md
---
## Remove `async` function&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiYXdhaXQgJCQkQ0FMTCIsInJld3JpdGUiOiIkJCRDQUxMICIsImNvbmZpZyI6ImlkOiByZW1vdmUtYXN5bmMtZGVmXG5sYW5ndWFnZTogcHl0aG9uXG5ydWxlOlxuICBwYXR0ZXJuOlxuICAgIGNvbnRleHQ6ICdhc3luYyBkZWYgJEZVTkMoJCQkQVJHUyk6ICQkJEJPRFknXG4gICAgc2VsZWN0b3I6IGZ1bmN0aW9uX2RlZmluaXRpb25cbnRyYW5zZm9ybTpcbiAgUkVNT1ZFRF9CT0RZOlxuICAgIHJld3JpdGU6XG4gICAgICByZXdyaXRlcnM6IFtyZW1vdmUtYXdhaXQtY2FsbF1cbiAgICAgIHNvdXJjZTogJCQkQk9EWVxuZml4OiB8LVxuICBkZWYgJEZVTkMoJCQkQVJHUyk6XG4gICAgJFJFTU9WRURfQk9EWVxucmV3cml0ZXJzOlxuLSBpZDogcmVtb3ZlLWF3YWl0LWNhbGxcbiAgcnVsZTpcbiAgICBwYXR0ZXJuOiAnYXdhaXQgJCQkQ0FMTCdcbiAgZml4OiAkJCRDQUxMXG4iLCJzb3VyY2UiOiJhc3luYyBkZWYgbWFpbjMoKTpcbiAgYXdhaXQgc29tZWNhbGwoMSwgNSkifQ==)

### Description

The `async` keyword in Python is used to define asynchronous functions that can be `await`ed.

In this example, we want to remove the `async` keyword from a function definition and replace it with a synchronous version of the function. We also need to remove the `await` keyword from the function body.

By default, ast-grep will not apply overlapping replacements. This means `await` keywords will not be modified because they are inside the async function body.

However, we can use the [`rewriter`](https://ast-grep.github.io/reference/yaml/rewriter.html) to apply changes inside the matched function body.

### YAML

```yaml
id: remove-async-def
language: python
rule:
  # match async function definition
  pattern:
    context: 'async def $FUNC($$$ARGS): $$$BODY'
    selector: function_definition
rewriters:
# define a rewriter to remove the await keyword
  remove-await-call:
    pattern: 'await $$$CALL'
    fix: $$$CALL # remove await keyword
# apply the rewriter to the function body
transform:
  REMOVED_BODY:
    rewrite:
      rewriters: [remove-await-call]
      source: $$$BODY
fix: |-
  def $FUNC($$$ARGS):
    $REMOVED_BODY
```

### Example

```python
async def main3():
  await somecall(1, 5)
```

### Diff

```python
async def main3(): # [!code --]
  await somecall(1, 5) # [!code --]
def main3(): # [!code ++]
  somecall(1, 5) # [!code ++]
```

### Contributed by

Inspired by the ast-grep issue [#1185](https://github.com/ast-grep/ast-grep/issues/1185)

---

---
url: /catalog/c/match-function-call.md
---
## Match Function Call in C

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImMiLCJxdWVyeSI6InRlc3QoJCQkKSIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBcbiAgICBjb250ZXh0OiAkTSgkJCQpO1xuICAgIHNlbGVjdG9yOiBjYWxsX2V4cHJlc3Npb24iLCJzb3VyY2UiOiIjZGVmaW5lIHRlc3QoeCkgKDIqeClcbmludCBhID0gdGVzdCgyKTtcbmludCBtYWluKCl7XG4gICAgaW50IGIgPSB0ZXN0KDIpO1xufSJ9)

### Description

One of the common questions of ast-grep is to match function calls in C.

A plain pattern like `test($A)` will not work. This is because [tree-sitter-c](https://github.com/tree-sitter/tree-sitter-c)
parse the code snippet into `macro_type_specifier`, see the [pattern output](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiYyIsInF1ZXJ5IjoidGVzdCgkJCQpIiwicmV3cml0ZSI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IFxuICAgIGNvbnRleHQ6ICRNKCQkJCk7XG4gICAgc2VsZWN0b3I6IGNhbGxfZXhwcmVzc2lvbiIsInNvdXJjZSI6IiNkZWZpbmUgdGVzdCh4KSAoMip4KVxuaW50IGEgPSB0ZXN0KDIpO1xuaW50IG1haW4oKXtcbiAgICBpbnQgYiA9IHRlc3QoMik7XG59In0=).

To avoid this ambiguity, ast-grep lets us write a [contextual pattern](/guide/rule-config/atomic-rule.html#pattern), which is a pattern inside a larger code snippet.
We can use `context` to write a pattern like this: `test($A);`. Then, we can use the selector `call_expression` to match only function calls.

### YAML

```yaml
id: match-function-call
language: c
rule:
  pattern:
    context: $M($$$);
    selector: call_expression
```

### Example

```c{2,4}
#define test(x) (2*x)
int a = test(2);
int main(){
    int b = test(2);
}
```

### Caveat

Note, tree-sitter-c parses code differently when it receives code fragment. For example,

* `test(a)` is parsed as `macro_type_specifier`
* `test(a);` is parsed as `expression_statement -> call_expression`
* `int b = test(a)` is parsed as `declaration -> init_declarator -> call_expression`

The behavior is controlled by how the tree-sitter parser is written. And tree-sitter-c behaves differently from [tree-sitter-cpp](https://github.com/tree-sitter/tree-sitter-cpp).

Please file issues on tree-sitter-c repo if you want to change the behavior. ast-grep will respect changes and decision from upstream authors.

---

---
url: /catalog/html/upgrade-ant-design-vue.md
---
## Upgrade Ant Design Vue&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6Imh0bWwiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoidXRpbHM6XG4gIGluc2lkZS10YWc6XG4gICAgaW5zaWRlOlxuICAgICAga2luZDogZWxlbWVudCBcbiAgICAgIHN0b3BCeTogeyBraW5kOiBlbGVtZW50IH1cbiAgICAgIGhhczpcbiAgICAgICAgc3RvcEJ5OiB7IGtpbmQ6IHRhZ19uYW1lIH1cbiAgICAgICAga2luZDogdGFnX25hbWVcbiAgICAgICAgcGF0dGVybjogJFRBR19OQU1FXG5ydWxlOlxuICBraW5kOiBhdHRyaWJ1dGVfbmFtZVxuICByZWdleDogOnZpc2libGVcbiAgbWF0Y2hlczogaW5zaWRlLXRhZyAgXG5maXg6IDpvcGVuXG5jb25zdHJhaW50czpcbiAgVEFHX05BTUU6XG4gICAgcmVnZXg6IGEtbW9kYWx8YS10b29sdGlwIiwic291cmNlIjoiPHRlbXBsYXRlPlxuICA8YS1tb2RhbCA6dmlzaWJsZT1cInZpc2libGVcIj5jb250ZW50PC9hLW1vZGFsPlxuICA8YS10b29sdGlwIDp2aXNpYmxlPVwidmlzaWJsZVwiIC8+XG4gIDxhLXRhZyA6dmlzaWJsZT1cInZpc2libGVcIj50YWc8L2EtdGFnPlxuPC90ZW1wbGF0ZT4ifQ==)

### Description

ast-grep can be used to upgrade Vue template using the HTML parser.

This rule is an example to upgrade [one breaking change](https://next.antdv.com/docs/vue/migration-v4#component-api-adjustment) in [Ant Design Vue](https://next.antdv.com/components/overview) from v3 to v4, unifying the controlled visible API of the component popup.

It is designed to identify and replace the `visible` attribute with the `open` attribute for specific components like `a-modal` and `a-tooltip`. Note the rule should not replace other visible attributes that are not related to the component popup like `a-tag`.

The rule can be broken down into the following steps:

1. Find the target attribute name by `kind` and `regex`
2. Find the attribute's enclosing element using `inside`, and get its tag name
3. Ensure the tag name is related to popup components, using constraints

### YAML

```yaml
id: upgrade-ant-design-vue
language: HTML
utils:
  inside-tag:
    # find the enclosing element of the attribute
    inside:
      kind: element
      stopBy: { kind: element } # only the closest element
      # find the tag name and store it in metavar
      has:
        stopBy: { kind: tag_name }
        kind: tag_name
        pattern: $TAG_NAME
rule:
  # find the target attribute_name
  kind: attribute_name
  regex: :visible
  # find the element
  matches: inside-tag
# ensure it only matches modal/tooltip but not tag
constraints:
  TAG_NAME:
    regex: a-modal|a-tooltip
fix: :open
```

### Example

```html {2,3}
<template>
  <a-modal :visible="visible">content</a-modal>
  <a-tooltip :visible="visible" />
  <a-tag :visible="visible">tag</a-tag>
</template>
```

### Diff

```html
<template>
  <a-modal :visible="visible">content</a-modal> // [!code --]
  <a-modal :open="visible">content</a-modal> // [!code ++]
  <a-tooltip :visible="visible" /> // [!code --]
  <a-tooltip :open="visible" /> // [!code ++]
  <a-tag :visible="visible">tag</a-tag>
</template>
```

### Contributed by

Inspired by [Vue.js RFC](https://github.com/vuejs/rfcs/discussions/705#discussion-7255672)

---

---
url: /catalog/typescript/speed-up-barrel-import.md
---
## Speed up Barrel Import&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJjb25maWciOiJydWxlOlxuICBwYXR0ZXJuOiBpbXBvcnQgeyQkJElERU5UU30gZnJvbSAnLi9iYXJyZWwnXG5yZXdyaXRlcnM6XG4tIGlkOiByZXdyaXRlLWlkZW50aWZlclxuICBydWxlOlxuICAgIHBhdHRlcm46ICRJREVOVFxuICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgZml4OiBpbXBvcnQgJElERU5UIGZyb20gJy4vYmFycmVsLyRJREVOVCdcbnRyYW5zZm9ybTpcbiAgSU1QT1JUUzpcbiAgICByZXdyaXRlOlxuICAgICAgcmV3cml0ZXJzOiBbcmV3cml0ZS1pZGVudGlmZXJdXG4gICAgICBzb3VyY2U6ICQkJElERU5UU1xuICAgICAgam9pbkJ5OiBcIlxcblwiXG5maXg6ICRJTVBPUlRTIiwic291cmNlIjoiaW1wb3J0IHsgYSwgYiwgYyB9IGZyb20gJy4vYmFycmVsJzsifQ==)

### Description

A [barrel import](https://adrianfaciu.dev/posts/barrel-files/) is a way to consolidate the exports of multiple modules into a single convenient module that can be imported using a single import statement. For instance, `import {a, b, c} from './barrel'`.

It has [some](https://vercel.com/blog/how-we-optimized-package-imports-in-next-js) [benefits](https://marvinh.dev/blog/speeding-up-javascript-ecosystem-part-7/) to import each module directly from its own file without going through the barrel file.
Such as reducing [bundle size](https://dev.to/tassiofront/barrel-files-and-why-you-should-stop-using-them-now-bc4), improving building time or avoiding [conflicting names](https://flaming.codes/posts/barrel-files-in-javascript/).

### YAML

```yaml
id: speed-up-barrel-import
language: typescript
# find the barrel import statement
rule:
  pattern: import {$$$IDENTS} from './barrel'
# rewrite imported identifiers to direct imports
rewriters:
- id: rewrite-identifer
  rule:
    pattern: $IDENT
    kind: identifier
  fix: import $IDENT from './barrel/$IDENT'
# apply the rewriter to the import statement
transform:
  IMPORTS:
    rewrite:
      rewriters: [rewrite-identifer]
      # $$$IDENTS contains imported identifiers
      source: $$$IDENTS
      # join the rewritten imports by newline
      joinBy: "\n"
fix: $IMPORTS
```

### Example

```ts {1}
import {a, b, c} from './barrel'
```

### Diff

```ts
import {a, b, c} from './barrel' // [!code --]
import a from './barrel/a' // [!code ++]
import b from './barrel/b' // [!code ++]
import c from './barrel/c' // [!code ++]
```

### Contributed by

[Herrington Darkholme](https://x.com/hd_nvim)

---

---
url: /catalog/typescript/find-import-identifiers.md
---
## Find Import Identifiers

* [Playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiIjIGZpbmQtYWxsLWltcG9ydHMtYW5kLXJlcXVpcmVzLnlhbWxcbmlkOiBmaW5kLWFsbC1pbXBvcnRzLWFuZC1yZXF1aXJlc1xubGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbm1lc3NhZ2U6IEZvdW5kIG1vZHVsZSBpbXBvcnQgb3IgcmVxdWlyZS5cbnNldmVyaXR5OiBpbmZvXG5ydWxlOlxuICBhbnk6XG4gICAgIyBBTElBUyBJTVBPUlRTXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGltcG9ydCB7IE9SSUdJTkFMIGFzIEFMSUFTIH0gZnJvbSAnU09VUkNFJ1xuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgICMgMS4gVGFyZ2V0IHRoZSBzcGVjaWZpYyBub2RlIHR5cGUgZm9yIG5hbWVkIGltcG9ydHNcbiAgICAgICAgLSBraW5kOiBpbXBvcnRfc3BlY2lmaWVyXG4gICAgICAgICMgMi4gRW5zdXJlIGl0ICpoYXMqIGFuICdhbGlhcycgZmllbGQsIGNhcHR1cmluZyB0aGUgYWxpYXMgaWRlbnRpZmllclxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBhbGlhc1xuICAgICAgICAgICAgcGF0dGVybjogJEFMSUFTXG4gICAgICAgICMgMy4gQ2FwdHVyZSB0aGUgb3JpZ2luYWwgaWRlbnRpZmllciAod2hpY2ggaGFzIHRoZSAnbmFtZScgZmllbGQpXG4gICAgICAgIC0gaGFzOlxuICAgICAgICAgICAgZmllbGQ6IG5hbWVcbiAgICAgICAgICAgIHBhdHRlcm46ICRPUklHSU5BTFxuICAgICAgICAjIDQuIEZpbmQgYW4gQU5DRVNUT1IgaW1wb3J0X3N0YXRlbWVudCBhbmQgY2FwdHVyZSBpdHMgc291cmNlIHBhdGhcbiAgICAgICAgLSBpbnNpZGU6XG4gICAgICAgICAgICBzdG9wQnk6IGVuZCAjIDw8PC0tLSBUaGlzIGlzIHRoZSBrZXkgZml4ISBTZWFyY2ggYW5jZXN0b3JzLlxuICAgICAgICAgICAga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgICAgICAgaGFzOiAjIEVuc3VyZSB0aGUgZm91bmQgaW1wb3J0X3N0YXRlbWVudCBoYXMgdGhlIHNvdXJjZSBmaWVsZFxuICAgICAgICAgICAgICBmaWVsZDogc291cmNlXG4gICAgICAgICAgICAgIHBhdHRlcm46ICRTT1VSQ0VcblxuICAgICMgREVGQVVMVCBJTVBPUlRTXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGltcG9ydCB7IE9SSUdJTkFMIH0gZnJvbSAnU09VUkNFJ1xuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgIC0ga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgICMgRW5zdXJlIGl0IGhhcyBhbiBpbXBvcnRfY2xhdXNlLi4uXG4gICAgICAgICAgICBraW5kOiBpbXBvcnRfY2xhdXNlXG4gICAgICAgICAgICBoYXM6XG4gICAgICAgICAgICAgICMgLi4udGhhdCBkaXJlY3RseSBjb250YWlucyBhbiBpZGVudGlmaWVyICh0aGUgZGVmYXVsdCBpbXBvcnQgbmFtZSlcbiAgICAgICAgICAgICAgIyBUaGlzIGlkZW50aWZpZXIgaXMgTk9UIHVuZGVyIGEgJ25hbWVkX2ltcG9ydHMnIG9yICduYW1lc3BhY2VfaW1wb3J0JyBub2RlXG4gICAgICAgICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgICAgICAgcGF0dGVybjogJERFRkFVTFRfTkFNRVxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBzb3VyY2VcbiAgICAgICAgICAgIHBhdHRlcm46ICRTT1VSQ0VcbiAgICBcbiAgICAjIFJFR1VMQVIgSU1QT1JUU1xuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgIyBpbXBvcnQgeyBPUklHSU5BTCB9IGZyb20gJ1NPVVJDRSdcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgIC0gYWxsOlxuICAgICAgICAjIDEuIFRhcmdldCB0aGUgc3BlY2lmaWMgbm9kZSB0eXBlIGZvciBuYW1lZCBpbXBvcnRzXG4gICAgICAgIC0ga2luZDogaW1wb3J0X3NwZWNpZmllclxuICAgICAgICAjIDIuIEVuc3VyZSBpdCAqaGFzKiBhbiAnYWxpYXMnIGZpZWxkLCBjYXB0dXJpbmcgdGhlIGFsaWFzIGlkZW50aWZpZXJcbiAgICAgICAgLSBoYXM6XG4gICAgICAgICAgICBmaWVsZDogbmFtZVxuICAgICAgICAgICAgcGF0dGVybjogJE9SSUdJTkFMXG4gICAgICAgICMgNC4gRmluZCBhbiBBTkNFU1RPUiBpbXBvcnRfc3RhdGVtZW50IGFuZCBjYXB0dXJlIGl0cyBzb3VyY2UgcGF0aFxuICAgICAgICAtIGluc2lkZTpcbiAgICAgICAgICAgIHN0b3BCeTogZW5kICMgPDw8LS0tIFRoaXMgaXMgdGhlIGtleSBmaXghIFNlYXJjaCBhbmNlc3RvcnMuXG4gICAgICAgICAgICBraW5kOiBpbXBvcnRfc3RhdGVtZW50XG4gICAgICAgICAgICBoYXM6ICMgRW5zdXJlIHRoZSBmb3VuZCBpbXBvcnRfc3RhdGVtZW50IGhhcyB0aGUgc291cmNlIGZpZWxkXG4gICAgICAgICAgICAgIGZpZWxkOiBzb3VyY2VcbiAgICAgICAgICAgICAgcGF0dGVybjogJFNPVVJDRVxuXG4gICAgIyBEWU5BTUlDIElNUE9SVFMgKFNpbmdsZSBWYXJpYWJsZSBBc3NpZ25tZW50KSBcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgICMgZWc6IChjb25zdCBWQVJfTkFNRSA9IHJlcXVpcmUoJ1NPVVJDRScpKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgIC0ga2luZDogdmFyaWFibGVfZGVjbGFyYXRvclxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBuYW1lXG4gICAgICAgICAgICBraW5kOiBpZGVudGlmaWVyXG4gICAgICAgICAgICBwYXR0ZXJuOiAkVkFSX05BTUUgIyBDYXB0dXJlIHRoZSBzaW5nbGUgdmFyaWFibGUgbmFtZVxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiB2YWx1ZVxuICAgICAgICAgICAgYW55OlxuICAgICAgICAgICAgICAjIERpcmVjdCBjYWxsXG4gICAgICAgICAgICAgIC0gYWxsOiAjIFdyYXAgY29uZGl0aW9ucyBpbiBhbGxcbiAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogZnVuY3Rpb24sIHJlZ2V4OiAnXihyZXF1aXJlfGltcG9ydCkkJyB9XG4gICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogYXJndW1lbnRzLCBoYXM6IHsga2luZDogc3RyaW5nLCBwYXR0ZXJuOiAkU09VUkNFIH0gfSAjIENhcHR1cmUgc291cmNlXG4gICAgICAgICAgICAgICMgQXdhaXRlZCBjYWxsXG4gICAgICAgICAgICAgIC0ga2luZDogYXdhaXRfZXhwcmVzc2lvblxuICAgICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICAgIGFsbDogIyBXcmFwIGNvbmRpdGlvbnMgaW4gYWxsXG4gICAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgIC0gaGFzOiB7IGZpZWxkOiBmdW5jdGlvbiwgcmVnZXg6ICdeKHJlcXVpcmV8aW1wb3J0KSQnIH1cbiAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGFyZ3VtZW50cywgaGFzOiB7IGtpbmQ6IHN0cmluZywgcGF0dGVybjogJFNPVVJDRSB9IH0gIyBDYXB0dXJlIHNvdXJjZVxuXG4gICAgIyBEWU5BTUlDIElNUE9SVFMgKERlc3RydWN0dXJlZCBTaG9ydGhhbmQgQXNzaWdubWVudCkgICAgIFxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgIyBlZzogKGNvbnN0IHsgT1JJR0lOQUwgfSA9IHJlcXVpcmUoJ1NPVVJDRScpKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgICMgMS4gVGFyZ2V0IHRoZSBzaG9ydGhhbmQgaWRlbnRpZmllciB3aXRoaW4gdGhlIHBhdHRlcm5cbiAgICAgICAgLSBraW5kOiBzaG9ydGhhbmRfcHJvcGVydHlfaWRlbnRpZmllcl9wYXR0ZXJuXG4gICAgICAgIC0gcGF0dGVybjogJE9SSUdJTkFMXG4gICAgICAgICMgMi4gRW5zdXJlIGl0J3MgaW5zaWRlIGFuIG9iamVjdF9wYXR0ZXJuIHRoYXQgaXMgdGhlIG5hbWUgb2YgYSB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgIC0gaW5zaWRlOlxuICAgICAgICAgICAga2luZDogb2JqZWN0X3BhdHRlcm5cbiAgICAgICAgICAgIGluc2lkZTogIyBDaGVjayB0aGUgdmFyaWFibGVfZGVjbGFyYXRvciBpdCBiZWxvbmdzIHRvXG4gICAgICAgICAgICAgIGtpbmQ6IHZhcmlhYmxlX2RlY2xhcmF0b3JcbiAgICAgICAgICAgICAgIyAzLiBDaGVjayB0aGUgdmFsdWUgYXNzaWduZWQgYnkgdGhlIHZhcmlhYmxlX2RlY2xhcmF0b3JcbiAgICAgICAgICAgICAgaGFzOlxuICAgICAgICAgICAgICAgIGZpZWxkOiB2YWx1ZVxuICAgICAgICAgICAgICAgIGFueTpcbiAgICAgICAgICAgICAgICAgICMgRGlyZWN0IGNhbGxcbiAgICAgICAgICAgICAgICAgIC0gYWxsOlxuICAgICAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGZ1bmN0aW9uLCByZWdleDogJ14ocmVxdWlyZXxpbXBvcnQpJCcgfVxuICAgICAgICAgICAgICAgICAgICAgIC0gaGFzOiB7IGZpZWxkOiBhcmd1bWVudHMsIGhhczogeyBraW5kOiBzdHJpbmcsIHBhdHRlcm46ICRTT1VSQ0UgfSB9ICMgQ2FwdHVyZSBzb3VyY2VcbiAgICAgICAgICAgICAgICAgICMgQXdhaXRlZCBjYWxsXG4gICAgICAgICAgICAgICAgICAtIGtpbmQ6IGF3YWl0X2V4cHJlc3Npb25cbiAgICAgICAgICAgICAgICAgICAgaGFzOlxuICAgICAgICAgICAgICAgICAgICAgIGFsbDpcbiAgICAgICAgICAgICAgICAgICAgICAgIC0ga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogZnVuY3Rpb24sIHJlZ2V4OiAnXihyZXF1aXJlfGltcG9ydCkkJyB9XG4gICAgICAgICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogYXJndW1lbnRzLCBoYXM6IHsga2luZDogc3RyaW5nLCBwYXR0ZXJuOiAkU09VUkNFIH0gfSAjIENhcHR1cmUgc291cmNlXG4gICAgICAgICAgICAgIHN0b3BCeTogZW5kICMgU2VhcmNoIGFuY2VzdG9ycyB0byBmaW5kIHRoZSBjb3JyZWN0IHZhcmlhYmxlX2RlY2xhcmF0b3JcblxuICAgICMgRFlOQU1JQyBJTVBPUlRTIChEZXN0cnVjdHVyZWQgQWxpYXMgQXNzaWdubWVudCkgXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGVnOiAoY29uc3QgeyBPUklHSU5BTDogQUxJQVMgfSA9IHJlcXVpcmUoJ1NPVVJDRScpKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgICMgMS4gVGFyZ2V0IHRoZSBwYWlyX3BhdHRlcm4gZm9yIGFsaWFzZWQgZGVzdHJ1Y3R1cmluZ1xuICAgICAgICAtIGtpbmQ6IHBhaXJfcGF0dGVyblxuICAgICAgICAjIDIuIENhcHR1cmUgdGhlIG9yaWdpbmFsIGlkZW50aWZpZXIgKGtleSlcbiAgICAgICAgLSBoYXM6XG4gICAgICAgICAgICBmaWVsZDoga2V5XG4gICAgICAgICAgICBraW5kOiBwcm9wZXJ0eV9pZGVudGlmaWVyICMgQ291bGQgYmUgc3RyaW5nL251bWJlciBsaXRlcmFsIHRvbywgYnV0IHByb3BlcnR5X2lkZW50aWZpZXIgaXMgY29tbW9uXG4gICAgICAgICAgICBwYXR0ZXJuOiAkT1JJR0lOQUxcbiAgICAgICAgIyAzLiBDYXB0dXJlIHRoZSBhbGlhcyBpZGVudGlmaWVyICh2YWx1ZSlcbiAgICAgICAgLSBoYXM6XG4gICAgICAgICAgICBmaWVsZDogdmFsdWVcbiAgICAgICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgICAgIHBhdHRlcm46ICRBTElBU1xuICAgICAgICAjIDQuIEVuc3VyZSBpdCdzIGluc2lkZSBhbiBvYmplY3RfcGF0dGVybiB0aGF0IGlzIHRoZSBuYW1lIG9mIGEgdmFyaWFibGVfZGVjbGFyYXRvclxuICAgICAgICAtIGluc2lkZTpcbiAgICAgICAgICAgIGtpbmQ6IG9iamVjdF9wYXR0ZXJuXG4gICAgICAgICAgICBpbnNpZGU6ICMgQ2hlY2sgdGhlIHZhcmlhYmxlX2RlY2xhcmF0b3IgaXQgYmVsb25ncyB0b1xuICAgICAgICAgICAgICBraW5kOiB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgICAgICAgICMgNS4gQ2hlY2sgdGhlIHZhbHVlIGFzc2lnbmVkIGJ5IHRoZSB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICBmaWVsZDogdmFsdWVcbiAgICAgICAgICAgICAgICBhbnk6XG4gICAgICAgICAgICAgICAgICAjIERpcmVjdCBjYWxsXG4gICAgICAgICAgICAgICAgICAtIGFsbDpcbiAgICAgICAgICAgICAgICAgICAgICAtIGtpbmQ6IGNhbGxfZXhwcmVzc2lvblxuICAgICAgICAgICAgICAgICAgICAgIC0gaGFzOiB7IGZpZWxkOiBmdW5jdGlvbiwgcmVnZXg6ICdeKHJlcXVpcmV8aW1wb3J0KSQnIH1cbiAgICAgICAgICAgICAgICAgICAgICAtIGhhczogeyBmaWVsZDogYXJndW1lbnRzLCBoYXM6IHsga2luZDogc3RyaW5nLCBwYXR0ZXJuOiAkU09VUkNFIH0gfSAjIENhcHR1cmUgc291cmNlXG4gICAgICAgICAgICAgICAgICAjIEF3YWl0ZWQgY2FsbFxuICAgICAgICAgICAgICAgICAgLSBraW5kOiBhd2FpdF9leHByZXNzaW9uXG4gICAgICAgICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICAgICAgICBhbGw6XG4gICAgICAgICAgICAgICAgICAgICAgICAtIGtpbmQ6IGNhbGxfZXhwcmVzc2lvblxuICAgICAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGZ1bmN0aW9uLCByZWdleDogJ14ocmVxdWlyZXxpbXBvcnQpJCcgfVxuICAgICAgICAgICAgICAgICAgICAgICAgLSBoYXM6IHsgZmllbGQ6IGFyZ3VtZW50cywgaGFzOiB7IGtpbmQ6IHN0cmluZywgcGF0dGVybjogJFNPVVJDRSB9IH0gIyBDYXB0dXJlIHNvdXJjZVxuICAgICAgICAgICAgICBzdG9wQnk6IGVuZCAjIFNlYXJjaCBhbmNlc3RvcnMgdG8gZmluZCB0aGUgY29ycmVjdCB2YXJpYWJsZV9kZWNsYXJhdG9yXG4gICAgICAgICAgICBzdG9wQnk6IGVuZCAjIEVuc3VyZSB3ZSBjaGVjayBhbmNlc3RvcnMgZm9yIHRoZSB2YXJpYWJsZV9kZWNsYXJhdG9yXG5cbiAgICAjIERZTkFNSUMgSU1QT1JUUyAoU2lkZSBFZmZlY3QgLyBTb3VyY2UgT25seSkgXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAjIGVnOiAocmVxdWlyZSgnU09VUkNFJykpXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAtIGFsbDpcbiAgICAgICAgLSBraW5kOiBzdHJpbmcgIyBUYXJnZXQgdGhlIHNvdXJjZSBzdHJpbmcgbGl0ZXJhbCBkaXJlY3RseVxuICAgICAgICAtIHBhdHRlcm46ICRTT1VSQ0VcbiAgICAgICAgLSBpbnNpZGU6ICMgU3RyaW5nIG11c3QgYmUgdGhlIGFyZ3VtZW50IG9mIHJlcXVpcmUoKSBvciBpbXBvcnQoKVxuICAgICAgICAgICAga2luZDogYXJndW1lbnRzXG4gICAgICAgICAgICBwYXJlbnQ6XG4gICAgICAgICAgICAgIGtpbmQ6IGNhbGxfZXhwcmVzc2lvblxuICAgICAgICAgICAgICBoYXM6XG4gICAgICAgICAgICAgICAgZmllbGQ6IGZ1bmN0aW9uXG4gICAgICAgICAgICAgICAgIyBNYXRjaCAncmVxdWlyZScgaWRlbnRpZmllciBvciAnaW1wb3J0JyBrZXl3b3JkIHVzZWQgZHluYW1pY2FsbHlcbiAgICAgICAgICAgICAgICByZWdleDogJ14ocmVxdWlyZXxpbXBvcnQpJCdcbiAgICAgICAgICAgIHN0b3BCeTogZW5kICMgU2VhcmNoIGFuY2VzdG9ycyBpZiBuZWVkZWQgKGZvciB0aGUgYXJndW1lbnRzL2NhbGxfZXhwcmVzc2lvbilcbiAgICAgICAgLSBub3Q6XG4gICAgICAgICAgICBpbnNpZGU6XG4gICAgICAgICAgICAgIGtpbmQ6IGxleGljYWxfZGVjbGFyYXRpb25cbiAgICAgICAgICAgICAgc3RvcEJ5OiBlbmQgIyBTZWFyY2ggYWxsIGFuY2VzdG9ycyB1cCB0byB0aGUgcm9vdFxuXG4gICAgIyBOQU1FU1BBQ0UgSU1QT1JUUyBcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgICMgZWc6IChpbXBvcnQgKiBhcyBucyBmcm9tICdtb2QnKVxuICAgICMgLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tXG4gICAgLSBhbGw6XG4gICAgICAgIC0ga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGtpbmQ6IGltcG9ydF9jbGF1c2VcbiAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAga2luZDogbmFtZXNwYWNlX2ltcG9ydFxuICAgICAgICAgICAgICBoYXM6XG4gICAgICAgICAgICAgICAgIyBuYW1lc3BhY2VfaW1wb3J0J3MgY2hpbGQgaWRlbnRpZmllciBpcyB0aGUgYWxpYXNcbiAgICAgICAgICAgICAgICBraW5kOiBpZGVudGlmaWVyXG4gICAgICAgICAgICAgICAgcGF0dGVybjogJE5BTUVTUEFDRV9BTElBU1xuICAgICAgICAtIGhhczpcbiAgICAgICAgICAgIGZpZWxkOiBzb3VyY2VcbiAgICAgICAgICAgIHBhdHRlcm46ICRTT1VSQ0VcblxuICAgICMgU0lERSBFRkZFQ1QgSU1QT1JUUyBcbiAgICAjIC0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLVxuICAgICMgZWc6IChpbXBvcnQgJ21vZCcpXG4gICAgIyAtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS1cbiAgICAtIGFsbDpcbiAgICAgICAgLSBraW5kOiBpbXBvcnRfc3RhdGVtZW50XG4gICAgICAgIC0gbm90OiAjIE11c3QgTk9UIGhhdmUgYW4gaW1wb3J0X2NsYXVzZVxuICAgICAgICAgICAgaGFzOiB7IGtpbmQ6IGltcG9ydF9jbGF1c2UgfVxuICAgICAgICAtIGhhczogIyBCdXQgbXVzdCBoYXZlIGEgc291cmNlXG4gICAgICAgICAgICBmaWVsZDogc291cmNlXG4gICAgICAgICAgICBwYXR0ZXJuOiAkU09VUkNFXG4iLCJzb3VyY2UiOiIvL0B0cy1ub2NoZWNrXG4vLyBOYW1lZCBpbXBvcnRcbmltcG9ydCB7IHRlc3RpbmcgfSBmcm9tICcuL3Rlc3RzJztcblxuLy8gQWxpYXNlZCBpbXBvcnRcbmltcG9ydCB7IHRlc3RpbmcgYXMgdGVzdCB9IGZyb20gJy4vdGVzdHMyJztcblxuLy8gRGVmYXVsdCBpbXBvcnRcbmltcG9ydCBoZWxsbyBmcm9tICdoZWxsb193b3JsZDEnO1xuXG4vLyBOYW1lc3BhY2UgaW1wb3J0XG5pbXBvcnQgKiBhcyBzb21ldGhpbmcgZnJvbSAnaGVsbG9fd29ybGQyJztcblxuLy8gU2lkZS1lZmZlY3QgaW1wb3J0XG5pbXBvcnQgJ0BmYXN0aWZ5L3N0YXRpYyc7XG5cbi8vIFR5cGUgaW1wb3J0XG5pbXBvcnQge3R5cGUgaGVsbG8xMjQzIGFzIHRlc3Rpbmd9IGZyb20gJ2hlbGxvJztcblxuLy8gUmVxdWlyZSBwYXR0ZXJuc1xuY29uc3QgbW9kID0gcmVxdWlyZSgnc29tZS1tb2R1bGUnKTtcbnJlcXVpcmUoJ3BvbHlmaWxsJyk7XG5cbi8vIERlc3RydWN0dXJlZCByZXF1aXJlXG5jb25zdCB7IHRlc3QxMjIsIHRlc3QyIH0gPSByZXF1aXJlKCcuL2Rlc3RydWN0dXJlZDEnKTtcbi8vIEFsaWFzZWQgcmVxdWlyZVxuY29uc3QgeyB0ZXN0MTIyOiB0ZXN0MTIzLCB0ZXN0MjogdGVzdDIzLCB0ZXN0MzogdGVzdDMzIH0gPSByZXF1aXJlKCcuL2Rlc3RydWN0dXJlZDInKTtcblxuLy8gTWl4ZWQgaW1wb3J0c1xuaW1wb3J0IGRlZmF1bHRFeHBvcnQsIHsgbmFtZWRFeHBvcnQgfSBmcm9tICcuL21peGVkJztcbmltcG9ydCBkZWZhdWx0RXhwb3J0MiwgKiBhcyBuYW1lc3BhY2UgZnJvbSAnLi9taXhlZDInO1xuXG5cbi8vIE11bHRpcGxlIGltcG9ydCBsaW5lcyBmcm9tIHRoZSBzYW1lIGZpbGVcbmltcG9ydCB7IG9uZSwgdHdvIGFzIGFsaWFzLCB0aHJlZSB9IGZyb20gJy4vbXVsdGlwbGUnO1xuaW1wb3J0IHsgbmV2ZXIsIGdvbm5hLCBnaXZlLCB5b3UsIHVwIH0gZnJvbSAnLi9tdWx0aXBsZSc7XG5cbi8vIFN0cmluZyBsaXRlcmFsIHZhcmlhdGlvbnNcbmltcG9ydCB7IHRlc3QxIH0gZnJvbSBcIi4vZG91YmxlLXF1b3RlZFwiO1xuaW1wb3J0IHsgdGVzdDIgfSBmcm9tICcuL3NpbmdsZS1xdW90ZWQnO1xuXG4vLyBNdWx0aWxpbmUgaW1wb3J0c1xuaW1wb3J0IHtcbiAgICBsb25nSW1wb3J0MSxcbiAgICBsb25nSW1wb3J0MiBhcyBhbGlhczIsXG4gICAgbG9uZ0ltcG9ydDNcbn0gZnJvbSAnLi9tdWx0aWxpbmUnO1xuXG4vLyBEeW5hbWljIGltcG9ydHNcbmNvbnN0IGR5bmFtaWNNb2R1bGUgPSBpbXBvcnQoJy4vZHluYW1pYzEnKTtcbmNvbnN0IHt0ZXN0aW5nLCB0ZXN0aW5nMTIzfSA9IGltcG9ydCgnLi9keW5hbWljMicpO1xuY29uc3QgYXN5bmNEeW5hbWljTW9kdWxlID0gYXdhaXQgaW1wb3J0KCcuL2FzeW5jX2R5bmFtaWMxJykudGhlbihtb2R1bGUgPT4gbW9kdWxlLmRlZmF1bHQpO1xuLy8gQWxpYXNlZCBkeW5hbWljIGltcG9ydFxuY29uc3QgeyBvcmlnaW5hbElkZW50aWZpZXI6IGFsaWFzZWREeW5hbWljSW1wb3J0fSA9IGF3YWl0IGltcG9ydCgnLi9hc3luY19keW5hbWljMicpO1xuXG4vLyBDb21tZW50cyBpbiBpbXBvcnRzXG5pbXBvcnQgLyogdGVzdCAqLyB7IFxuICAgIC8vIENvbW1lbnQgaW4gaW1wb3J0XG4gICAgY29tbWVudGVkSW1wb3J0IFxufSBmcm9tICcuL2NvbW1lbnRlZCc7IC8vIEVuZCBvZiBsaW5lIGNvbW1lbnQgXG5cblxuIn0=)

### Description

Finding import metadata can be useful. Below is a comprehensive snippet for extracting identifiers from various import statements:

* Alias Imports (`import { hello as world } from './file'`)
* Default & Regular Imports (`import test from './my-test`')
* Dynamic Imports (`require(...)`, and `import(...)`)
* Side Effect & Namespace Imports (`import * as myCode from './code`')

### YAML

```yaml
# find-all-imports-and-identifiers.yaml
id: find-all-imports-and-identifiers
language: TypeScript
rule:
  any:
    # ALIAS IMPORTS
    # ------------------------------------------------------------
    # import { ORIGINAL as ALIAS } from 'SOURCE'
    # ------------------------------------------------------------
    - all:
        # 1. Target the specific node type for named imports
        - kind: import_specifier
        # 2. Ensure it *has* an 'alias' field, capturing the alias identifier
        - has:
            field: alias
            pattern: $ALIAS
        # 3. Capture the original identifier (which has the 'name' field)
        - has:
            field: name
            pattern: $ORIGINAL
        # 4. Find an ANCESTOR import_statement and capture its source path
        - inside:
            stopBy: end # <<<--- Search ancestors.
            kind: import_statement
            has: # Ensure the found import_statement has the source field
              field: source
              pattern: $SOURCE

    # DEFAULT IMPORTS
    # ------------------------------------------------------------
    # import { ORIGINAL } from 'SOURCE'
    # ------------------------------------------------------------
    - all:
        - kind: import_statement
        - has:
            # Ensure it has an import_clause...
            kind: import_clause
            has:
              # ...that directly contains an identifier (the default import name)
              # This identifier is NOT under a 'named_imports' or 'namespace_import' node
              kind: identifier
              pattern: $DEFAULT_NAME
        - has:
            field: source
            pattern: $SOURCE

    # REGULAR IMPORTS
    # ------------------------------------------------------------
    # import { ORIGINAL } from 'SOURCE'
    # ------------------------------------------------------------
    - all:
        # 1. Target the specific node type for named imports
        - kind: import_specifier
        # 2. Ensure it *has* an 'alias' field, capturing the alias identifier
        - has:
            field: name
            pattern: $ORIGINAL
        # 4. Find an ANCESTOR import_statement and capture its source path
        - inside:
            stopBy: end # <<<--- This is the key fix! Search ancestors.
            kind: import_statement
            has: # Ensure the found import_statement has the source field
              field: source
              pattern: $SOURCE

    # DYNAMIC IMPORTS (Single Variable Assignment)
    # ------------------------------------------------------------
    # const VAR_NAME = require('SOURCE')
    # ------------------------------------------------------------
    - all:
        - kind: variable_declarator
        - has:
            field: name
            kind: identifier
            pattern: $VAR_NAME # Capture the single variable name
        - has:
            field: value
            any:
              # Direct call
              - all: # Wrap conditions in all
                  - kind: call_expression
                  - has: { field: function, regex: '^(require|import)$' }
                  - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
              # Awaited call
              - kind: await_expression
                has:
                  all: # Wrap conditions in all
                    - kind: call_expression
                    - has: { field: function, regex: '^(require|import)$' }
                    - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source

    # DYNAMIC IMPORTS (Destructured Shorthand Assignment)
    # ------------------------------------------------------------
    # const { ORIGINAL } = require('SOURCE')
    # ------------------------------------------------------------
    - all:
        # 1. Target the shorthand identifier within the pattern
        - kind: shorthand_property_identifier_pattern
        - pattern: $ORIGINAL
        # 2. Ensure it's inside an object_pattern that is the name of a variable_declarator
        - inside:
            kind: object_pattern
            inside: # Check the variable_declarator it belongs to
              kind: variable_declarator
              # 3. Check the value assigned by the variable_declarator
              has:
                field: value
                any:
                  # Direct call
                  - all:
                      - kind: call_expression
                      - has: { field: function, regex: '^(require|import)$' }
                      - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
                  # Awaited call
                  - kind: await_expression
                    has:
                      all:
                        - kind: call_expression
                        - has: { field: function, regex: '^(require|import)$' }
                        - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
              stopBy: end # Search ancestors to find the correct variable_declarator

    # DYNAMIC IMPORTS (Destructured Alias Assignment)
    # ------------------------------------------------------------
    # const { ORIGINAL: ALIAS } = require('SOURCE')
    # ------------------------------------------------------------
    - all:
        # 1. Target the pair_pattern for aliased destructuring
        - kind: pair_pattern
        # 2. Capture the original identifier (key)
        - has:
            field: key
            kind: property_identifier # Could be string/number literal too, but property_identifier is common
            pattern: $ORIGINAL
        # 3. Capture the alias identifier (value)
        - has:
            field: value
            kind: identifier
            pattern: $ALIAS
        # 4. Ensure it's inside an object_pattern that is the name of a variable_declarator
        - inside:
            kind: object_pattern
            inside: # Check the variable_declarator it belongs to
              kind: variable_declarator
              # 5. Check the value assigned by the variable_declarator
              has:
                field: value
                any:
                  # Direct call
                  - all:
                      - kind: call_expression
                      - has: { field: function, regex: '^(require|import)$' }
                      - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
                  # Awaited call
                  - kind: await_expression
                    has:
                      all:
                        - kind: call_expression
                        - has: { field: function, regex: '^(require|import)$' }
                        - has: { field: arguments, has: { kind: string, pattern: $SOURCE } } # Capture source
              stopBy: end # Search ancestors to find the correct variable_declarator
            stopBy: end # Ensure we check ancestors for the variable_declarator

    # DYNAMIC IMPORTS (Side Effect / Source Only)
    # ------------------------------------------------------------
    # require('SOURCE')
    # ------------------------------------------------------------
    - all:
        - kind: string # Target the source string literal directly
        - pattern: $SOURCE
        - inside: # String must be the argument of require() or import()
            kind: arguments
            parent:
              kind: call_expression
              has:
                field: function
                # Match 'require' identifier or 'import' keyword used dynamically
                regex: '^(require|import)$'
            stopBy: end # Search ancestors if needed (for the arguments/call_expression)
        - not:
            inside:
              kind: lexical_declaration
              stopBy: end # Search all ancestors up to the root

    # NAMESPACE IMPORTS
    # ------------------------------------------------------------
    # import * as ns from 'mod'
    # ------------------------------------------------------------
    - all:
        - kind: import_statement
        - has:
            kind: import_clause
            has:
              kind: namespace_import
              has:
                # namespace_import's child identifier is the alias
                kind: identifier
                pattern: $NAMESPACE_ALIAS
        - has:
            field: source
            pattern: $SOURCE

    # SIDE EFFECT IMPORTS
    # ------------------------------------------------------------
    # import 'mod'
    # ------------------------------------------------------------
    - all:
        - kind: import_statement
        - not: # Must NOT have an import_clause
            has: { kind: import_clause }
        - has: # But must have a source
            field: source
            pattern: $SOURCE
```

### Example

```ts {60}
//@ts-nocheck
// Named import
import { testing } from './tests';

// Aliased import
import { testing as test } from './tests2';

// Default import
import hello from 'hello_world1';

// Namespace import
import * as something from 'hello_world2';

// Side-effect import
import '@fastify/static';

// Type import
import {type hello1243 as testing} from 'hello';

// Require patterns
const mod = require('some-module');
require('polyfill');

// Destructured require
const { test122, test2 } = require('./destructured1');
// Aliased require
const { test122: test123, test2: test23, test3: test33 } = require('./destructured2');

// Mixed imports
import defaultExport, { namedExport } from './mixed';
import defaultExport2, * as namespace from './mixed2';


// Multiple import lines from the same file
import { one, two as alias, three } from './multiple';
import { never, gonna, give, you, up } from './multiple';

// String literal variations
import { test1 } from "./double-quoted";
import { test2 } from './single-quoted';

// Multiline imports
import {
    longImport1,
    longImport2 as alias2,
    longImport3
} from './multiline';

// Dynamic imports
const dynamicModule = import('./dynamic1');
const {testing, testing123} = import('./dynamic2');
const asyncDynamicModule = await import('./async_dynamic1').then(module => module.default);
// Aliased dynamic import
const { originalIdentifier: aliasedDynamicImport} = await import('./async_dynamic2');

// Comments in imports
import /* test */ {
    // Comment in import
    commentedImport
} from './commented'; // End of line comment
```

### Contributed by

[Michael Angelo Rivera](https://github.com/michaelangeloio)

---

---
url: /catalog/rule-template.md
---
## Your Rule Name&#x20;

* [Playground Link](/playground.html#)

### Description

Some Description for your rule!

### Pattern

```shell
ast-grep -p pattern -r rewrite -l js
# or without fixer
ast-grep -p pattern -l js
```

### YAML

```yaml
```

### Example

```js {1}
var a = 123
```

### Diff

```js
var a = 123 // [!code --]
let a = 123 // [!code ++]
```

### Contributed by

[Author Name](https://your-social.link)

---

---
url: /catalog/ruby/detect-path-traversal.md
---
## Detect Path Traversal Vulnerability in Rails

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1YnkiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoiYXN0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogcGF0aC10cmF2ZXJzYWxcbm1lc3NhZ2U6IFBvdGVudGlhbCBQYXRoIFRyYXZlcnNhbCB2dWxuZXJhYmlsaXR5IGRldGVjdGVkLiBVc2VyIGlucHV0IGlzIGJlaW5nIHVzZWQgdG8gY29uc3RydWN0IGZpbGUgcGF0aHMgd2l0aG91dCB2YWxpZGF0aW9uLlxuc2V2ZXJpdHk6IGhpbnRcbmxhbmd1YWdlOiBSdWJ5XG5ub3RlOiB8XG4gIFBhdGggVHJhdmVyc2FsIChEaXJlY3RvcnkgVHJhdmVyc2FsKSBvY2N1cnMgd2hlbiB1c2VyIGlucHV0IGlzIHVzZWQgdG8gY29uc3RydWN0IGZpbGUgcGF0aHNcbiAgd2l0aG91dCBwcm9wZXIgdmFsaWRhdGlvbi4gVGhpcyBhbGxvd3MgYXR0YWNrZXJzIHRvIGFjY2VzcyBmaWxlcyBvdXRzaWRlIHRoZSBpbnRlbmRlZCBkaXJlY3RvcnkuXG4gIFZhbGlkYXRlIGFuZCBzYW5pdGl6ZSBmaWxlIHBhdGhzLCBhbmQgdXNlIEZpbGUuYmFzZW5hbWUoKSBvciBzaW1pbGFyIGZ1bmN0aW9ucy5cblxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogUmFpbHMucm9vdC5qb2luKCQkJCwgJFZBUiwgJCQkKVxuICAgIC0gcGF0dGVybjogRmlsZS5qb2luKCQkJCwgJFZBUiwgJCQkKVxuICAgIC0gcGF0dGVybjogc2VuZF9maWxlICRWQVIiLCJzb3VyY2UiOiIjIOaknOWHuuOBleOCjOOCi+OCs+ODvOODieS+i1xuIyDjg5Hjgr/jg7zjg7MxOiBSYWlscy5yb290LmpvaW4gd2l0aCB2YXJpYWJsZVxuUmFpbHMucm9vdC5qb2luKCd1cGxvYWRzJywgcGFyYW1zWzpmaWxlbmFtZV0pXG5SYWlscy5yb290LmpvaW4oJ2RhdGEnLCB1c2VyX2lucHV0LCAnZmlsZS50eHQnKVxuXG4jIOODkeOCv+ODvOODszI6IEZpbGUuam9pbiB3aXRoIHZhcmlhYmxlXG5GaWxlLmpvaW4oJy92YXIvd3d3JywgcGFyYW1zWzpwYXRoXSlcbkZpbGUuam9pbihiYXNlX3BhdGgsIHVzZXJfaWQsIGZpbGVuYW1lKVxuXG4jIOODkeOCv+ODvOODszM6IHNlbmRfZmlsZSB3aXRoIHZhcmlhYmxlXG5zZW5kX2ZpbGUgcGFyYW1zWzpmaWxlXVxuc2VuZF9maWxlIHVzZXIuZG9jdW1lbnRfcGF0aCJ9)

### Description

Path Traversal (Directory Traversal) occurs when user input is used to construct file paths without proper validation. This allows attackers to access files outside the intended directory by using special characters like `../` to navigate the filesystem.

This rule detects common path traversal patterns in Rails applications where user-controlled variables are used in:

* `Rails.root.join()` - Building file paths relative to the Rails application root
* `File.join()` - Constructing file paths
* `send_file` - Sending files to users

To prevent path traversal vulnerabilities, always validate and sanitize file paths, use `File.basename()` to extract only the filename, or use allowlists for permitted files.

### YAML

```yaml
id: path-traversal
message: Potential Path Traversal vulnerability detected. User input is being used to construct file paths without validation.
severity: hint
language: Ruby
note: |
  Path Traversal (Directory Traversal) occurs when user input is used to construct file paths
  without proper validation. This allows attackers to access files outside the intended directory.
  Validate and sanitize file paths, and use File.basename() or similar functions.

rule:
  any:
    - pattern: Rails.root.join($$$, $VAR, $$$)
    - pattern: File.join($$$, $VAR, $$$)
    - pattern: send_file $VAR
```

### Example

```rb {2,3,6,7,10,11}
# Pattern 1: Rails.root.join with variable
Rails.root.join('uploads', params[:filename])
Rails.root.join('data', user_input, 'file.txt')

# Pattern 2: File.join with variable
File.join('/var/www', params[:path])
File.join(base_path, user_id, filename)

# Pattern 3: send_file with variable
send_file params[:file]
send_file user.document_path
```

### Contributed by

[sora fs0414](https://x.com/_fs0414) from this [blog post](https://fs0414.hatenablog.com/entry/2025/11/02/032114)

---

---
url: /catalog/python/recursive-rewrite-type.md
---
## Recursive Rewrite Type&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicmV3cml0ZXJzOlxyXG4tIGlkOiBvcHRpb25hbFxyXG4gIGxhbmd1YWdlOiBQeXRob25cclxuICBydWxlOlxyXG4gICAgYW55OlxyXG4gICAgLSBwYXR0ZXJuOlxyXG4gICAgICAgIGNvbnRleHQ6ICdhcmc6IE9wdGlvbmFsWyRUWVBFXSdcclxuICAgICAgICBzZWxlY3RvcjogZ2VuZXJpY190eXBlXHJcbiAgICAtIHBhdHRlcm46IE9wdGlvbmFsWyRUWVBFXVxyXG4gIHRyYW5zZm9ybTpcclxuICAgIE5UOlxyXG4gICAgICByZXdyaXRlOiBcclxuICAgICAgICByZXdyaXRlcnM6IFtvcHRpb25hbCwgdW5pb25zXVxyXG4gICAgICAgIHNvdXJjZTogJFRZUEVcclxuICBmaXg6ICROVCB8IE5vbmVcclxuLSBpZDogdW5pb25zXHJcbiAgbGFuZ3VhZ2U6IFB5dGhvblxyXG4gIHJ1bGU6XHJcbiAgICBwYXR0ZXJuOlxyXG4gICAgICBjb250ZXh0OiAnYTogVW5pb25bJCQkVFlQRVNdJ1xyXG4gICAgICBzZWxlY3RvcjogZ2VuZXJpY190eXBlXHJcbiAgdHJhbnNmb3JtOlxyXG4gICAgVU5JT05TOlxyXG4gICAgICByZXdyaXRlOlxyXG4gICAgICAgIHJld3JpdGVyczpcclxuICAgICAgICAgIC0gcmV3cml0ZS11bmlvbnNcclxuICAgICAgICBzb3VyY2U6ICQkJFRZUEVTXHJcbiAgICAgICAgam9pbkJ5OiBcIiB8IFwiXHJcbiAgZml4OiAkVU5JT05TXHJcbi0gaWQ6IHJld3JpdGUtdW5pb25zXHJcbiAgcnVsZTpcclxuICAgIHBhdHRlcm46ICRUWVBFXHJcbiAgICBraW5kOiB0eXBlXHJcbiAgdHJhbnNmb3JtOlxyXG4gICAgTlQ6XHJcbiAgICAgIHJld3JpdGU6IFxyXG4gICAgICAgIHJld3JpdGVyczogW29wdGlvbmFsLCB1bmlvbnNdXHJcbiAgICAgICAgc291cmNlOiAkVFlQRVxyXG4gIGZpeDogJE5UXHJcbnJ1bGU6XHJcbiAga2luZDogdHlwZVxyXG4gIHBhdHRlcm46ICRUUEVcclxudHJhbnNmb3JtOlxyXG4gIE5FV19UWVBFOlxyXG4gICAgcmV3cml0ZTogXHJcbiAgICAgIHJld3JpdGVyczogW29wdGlvbmFsLCB1bmlvbnNdXHJcbiAgICAgIHNvdXJjZTogJFRQRVxyXG5maXg6ICRORVdfVFlQRSIsInNvdXJjZSI6InJlc3VsdHM6ICBPcHRpb25hbFtVbmlvbltMaXN0W1VuaW9uW3N0ciwgZGljdF1dLCBzdHJdXVxuIn0=)

### Description

Suppose we want to transform Python's `Union[T1, T2]` to `T1 | T2` and `Optional[T]` to `T | None`.

By default, ast-grep will only fix the outermost node that matches a pattern and will not rewrite the inner AST nodes inside a match. This avoids unexpected rewriting or infinite rewriting loop.

So if you are using non-recursive rewriter like [this](https://github.com/ast-grep/ast-grep/discussions/1566#discussion-7401382), `Optional[Union[int, str]]` will only be converted to `Union[int, str] | None`. Note the inner `Union[int, str]` is not enabled. This is because the rewriter `optional` matches `Optional[$TYPE]` and rewrite it to `$TYPE | None`. The inner `$TYPE` is not processed.

However, we can apply `rewriters` to inner types recursively. Take the `optional` rewriter as an example, we need to apply rewriters, `optional` and `unions`, **recursively** to `$TYPE` and get a new variable `$NT`.

### YAML

```yml
id: recursive-rewrite-types
language: python
rewriters:
# rewrite Optional[T] to T | None
- id: optional
  rule:
    any:
    - pattern:
        context: 'arg: Optional[$TYPE]'
        selector: generic_type
    - pattern: Optional[$TYPE]
  # recursively apply rewriters to $TYPE
  transform:
    NT:
      rewrite:
        rewriters: [optional, unions]
        source: $TYPE
  # use the new variable $NT
  fix: $NT | None

# similar to Optional, rewrite Union[T1, T2] to T1 | T2
- id: unions
  language: Python
  rule:
    pattern:
      context: 'a: Union[$$$TYPES]'
      selector: generic_type
  transform:
    UNIONS:
      # rewrite all types inside $$$TYPES
      rewrite:
        rewriters: [ rewrite-unions ]
        source: $$$TYPES
        joinBy: " | "
  fix: $UNIONS
- id: rewrite-unions
  rule:
    pattern: $TYPE
    kind: type
  # recursive part
  transform:
    NT:
      rewrite:
        rewriters: [optional, unions]
        source: $TYPE
  fix: $NT

# find all types
rule:
  kind: type
  pattern: $TPE
# apply the recursive rewriters
transform:
  NEW_TYPE:
    rewrite:
      rewriters: [optional, unions]
      source: $TPE
# output
fix: $NEW_TYPE
```

### Example

```python
results:  Optional[Union[List[Union[str, dict]], str]]
```

### Diff

```python
results:  Optional[Union[List[Union[str, dict]], str]] # [!code --]
results:  List[str | dict] | str | None #[!code ++]
```

### Contributed by

Inspired by [steinuil](https://github.com/ast-grep/ast-grep/discussions/1566)

---

---
url: /catalog/python/rewrite-sqlalchemy-mapped-column.md
---
## Rewrite SQLAlchemy with Type Annotations&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiYShudWxsYWJsZT1UcnVlKSIsInJld3JpdGUiOiIxMjMiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6ImtleXdvcmRfYXJndW1lbnQiLCJjb25maWciOiJpZDogcmVtb3ZlLW51bGxhYmxlLWFyZ1xubGFuZ3VhZ2U6IHB5dGhvblxucnVsZTpcbiAgcGF0dGVybjogJFggPSBtYXBwZWRfY29sdW1uKCQkJEFSR1MpXG4gIGFueTpcbiAgICAtIHBhdHRlcm46ICRYID0gbWFwcGVkX2NvbHVtbigkJCRCRUZPUkUsIFN0cmluZywgJCQkTUlELCBudWxsYWJsZT1UcnVlLCAkJCRBRlRFUilcbiAgICAtIHBhdHRlcm46ICRYID0gbWFwcGVkX2NvbHVtbigkJCRCRUZPUkUsIFN0cmluZywgJCQkTUlELCBudWxsYWJsZT1UcnVlKVxucmV3cml0ZXJzOlxuLSBpZDogZmlsdGVyLXN0cmluZy1udWxsYWJsZVxuICBydWxlOlxuICAgIHBhdHRlcm46ICRBUkdcbiAgICBpbnNpZGU6XG4gICAgICBraW5kOiBhcmd1bWVudF9saXN0XG4gICAgYWxsOlxuICAgIC0gbm90OiBcbiAgICAgICAgcGF0dGVybjogU3RyaW5nXG4gICAgLSBub3Q6XG4gICAgICAgIHBhdHRlcm46XG4gICAgICAgICAgY29udGV4dDogYShudWxsYWJsZT1UcnVlKVxuICAgICAgICAgIHNlbGVjdG9yOiBrZXl3b3JkX2FyZ3VtZW50XG4gIGZpeDogJEFSR1xuXG50cmFuc2Zvcm06XG4gIE5FV0FSR1M6XG4gICAgcmV3cml0ZTpcbiAgICAgIHJld3JpdGVyczogW2ZpbHRlci1zdHJpbmctbnVsbGFibGVdXG4gICAgICBzb3VyY2U6ICQkJEFSR1NcbiAgICAgIGpvaW5CeTogJywgJ1xuZml4OiB8LVxuICAkWDogTWFwcGVkW3N0ciB8IE5vbmVdID0gbWFwcGVkX2NvbHVtbigkTkVXQVJHUykiLCJzb3VyY2UiOiJtZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIGRlZmF1bHQ9XCJoZWxsb1wiLCBudWxsYWJsZT1UcnVlKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIG51bGxhYmxlPVRydWUpXG5cbl9tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihcIm1lc3NhZ2VcIiwgU3RyaW5nLCBudWxsYWJsZT1UcnVlKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIG51bGxhYmxlPVRydWUsIHVuaXF1ZT1UcnVlKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihcbiAgU3RyaW5nLCBpbmRleD1UcnVlLCBudWxsYWJsZT1UcnVlLCB1bmlxdWU9VHJ1ZSlcblxuIyBTaG91bGQgbm90IGJlIHRyYW5zZm9ybWVkXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIGRlZmF1bHQ9XCJoZWxsb1wiKVxuXG5tZXNzYWdlID0gbWFwcGVkX2NvbHVtbihTdHJpbmcsIGRlZmF1bHQ9XCJoZWxsb1wiLCBudWxsYWJsZT1GYWxzZSlcblxubWVzc2FnZSA9IG1hcHBlZF9jb2x1bW4oSW50ZWdlciwgZGVmYXVsdD1cImhlbGxvXCIpIn0=)

### Description

[SQLAlchemy 2.0](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html) recommends using type annotations with `Mapped` type for modern declarative mapping. The `mapped_column()` construct can derive its configuration from [PEP 484](https://peps.python.org/pep-0484/) type annotations.

This rule helps migrate legacy SQLAlchemy code that explicitly uses `String` type and `nullable=True` to the modern type annotation approach using `Mapped[str | None]`.

The key technique demonstrated here is using **rewriters** to selectively filter arguments. The rewriter:

1. Matches each argument inside the `argument_list`
2. Excludes the `String` type argument
3. Excludes the `nullable=True` keyword argument
4. Keeps all other arguments

### YAML

```yaml
id: remove-nullable-arg
language: python
rule:
  pattern: $X = mapped_column($$$ARGS)
  any:
    - pattern: $X = mapped_column($$$BEFORE, String, $$$MID, nullable=True, $$$AFTER)
    - pattern: $X = mapped_column($$$BEFORE, String, $$$MID, nullable=True)
rewriters:
- id: filter-string-nullable
  rule:
    pattern: $ARG
    inside:
      kind: argument_list
    all:
    - not:
        pattern: String
    - not:
        pattern:
          context: a(nullable=True)
          selector: keyword_argument
  fix: $ARG

transform:
  NEWARGS:
    rewrite:
      rewriters: [filter-string-nullable]
      source: $$$ARGS
      joinBy: ', '
fix: |-
  $X: Mapped[str | None] = mapped_column($NEWARGS)
```

### Example

```python {1,3,5,7-8}
message = mapped_column(String, default="hello", nullable=True)

message = mapped_column(String, nullable=True)

_message = mapped_column("message", String, nullable=True)

message = mapped_column(String, nullable=True, unique=True)

message = mapped_column(
  String, index=True, nullable=True, unique=True)

# Should not be transformed
message = mapped_column(String, default="hello")

message = mapped_column(String, default="hello", nullable=False)

message = mapped_column(Integer, default="hello")
```

### Diff

```python
message = mapped_column(String, default="hello", nullable=True) # [!code --]
message: Mapped[str | None] = mapped_column(default="hello") # [!code ++]

message = mapped_column(String, nullable=True) # [!code --]
message: Mapped[str | None] = mapped_column() # [!code ++]

_message = mapped_column("message", String, nullable=True) # [!code --]
_message: Mapped[str | None] = mapped_column("message") # [!code ++]

message = mapped_column(String, nullable=True, unique=True) # [!code --]
message: Mapped[str | None] = mapped_column(unique=True) # [!code ++]

message = mapped_column( # [!code --]
  String, index=True, nullable=True, unique=True) # [!code --]
message: Mapped[str | None] = mapped_column( # [!code ++]
  index=True, unique=True) # [!code ++]
```

### Contributed by

Inspired by [discussion #2319](https://github.com/ast-grep/ast-grep/discussions/2319)

---

---
url: /links/roadmap.md
---
# TODO:

## Core

* \[x] Add replace
* \[x] Add find\_all
* \[x] Add metavar char customization
* \[x] Add per-language customization
* \[x] Add support for vec/sequence matcher
* \[x] View node in context
* \[x] implement iterative DFS mode
* \[ ] Investigate perf heuristic (e.g. match fixed-string)
* \[x] Group matching rules based on root pattern kind id
* \[ ] Remove unwrap usage and implement error handling

## Metavariable Matcher

* \[x] Regex
* \[x] Pattern
* \[x] Kind
* \[ ] Use CoW to optimize MetaVarEnv

## Operators/Combinators

* \[x] every / all
* \[x] either / any
* \[x] inside
* \[x] has
* \[x] follows
* \[x] precedes

## CLI

* \[x] match against files in directory recursively
* \[x] interactive mode
* \[x] as dry run mode (listing all rewrite)
* \[x] inplace edit mode
* \[x] no-color mode
* \[x] JSON output
* \[ ] execute remote rules

## Config

* \[x] support YAML config rule
* \[x] Add support for severity
* \[x] Add support for error message
* \[x] Add support for error labels
* \[x] Add support for fix

## Binding

* \[ ] NAPI binding
* \[x] WASM binding
* \[ ] Python binding

## Playground

* \[x] build a playground based on WASM binding
* \[x] build YAML config for WASM playground
* \[x] URL sharing
* \[x] add fix/rewrite

## LSP

* \[x] Add LSP command
* \[ ] implement LSP incremental
* \[ ] add code action

## Builtin Ruleset

* \[ ] Migrate some ESLint rule (or RSLint rule)

---

---
url: /catalog/c/rewrite-method-to-function-call.md
---
## Rewrite Method to Function Call&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImMiLCJxdWVyeSI6IiRDT1VOVCA9ICRcbiIsInJld3JpdGUiOiIiLCJjb25maWciOiJpZDogbWV0aG9kX3JlY2VpdmVyXG5ydWxlOlxuICBwYXR0ZXJuOiAkUi4kTUVUSE9EKCQkJEFSR1MpXG50cmFuc2Zvcm06XG4gIE1BWUJFX0NPTU1BOlxuICAgIHJlcGxhY2U6XG4gICAgICBzb3VyY2U6ICQkJEFSR1NcbiAgICAgIHJlcGxhY2U6ICdeLisnXG4gICAgICBieTogJywgJ1xuZml4OlxuICAkTUVUSE9EKCYkUiRNQVlCRV9DT01NQSQkJEFSR1MpXG4iLCJzb3VyY2UiOiJ2b2lkIHRlc3RfZnVuYygpIHtcbiAgICBzb21lX3N0cnVjdC0+ZmllbGQubWV0aG9kKCk7XG4gICAgc29tZV9zdHJ1Y3QtPmZpZWxkLm90aGVyX21ldGhvZCgxLCAyLCAzKTtcbn0ifQ==)

### Description

In C, there is no built-in support for object-oriented programming, but some programmers use structs and function pointers to simulate classes and methods. However, this style can have some drawbacks, such as:

* extra memory allocation and deallocation for the struct and the function pointer.
* indirection overhead when calling the function pointer.

A possible alternative is to use a plain function call with the struct pointer as the first argument.

### YAML

```yaml
id: method_receiver
language: c
rule:
  pattern: $R.$METHOD($$$ARGS)
transform:
  MAYBE_COMMA:
    replace:
      source: $$$ARGS
      replace: '^.+'
      by: ', '
fix:
  $METHOD(&$R$MAYBE_COMMA$$$ARGS)
```

### Example

```c {2-3}
void test_func() {
    some_struct->field.method();
    some_struct->field.other_method(1, 2, 3);
}
```

### Diff

```c
void test_func() {
    some_struct->field.method(); // [!code --]
    method(&some_struct->field); // [!code ++]
    some_struct->field.other_method(1, 2, 3); // [!code --]
    other_method(&some_struct->field, 1, 2, 3); // [!code ++]
}
```

### Contributed by

[Surma](https://twitter.com/DasSurma), adapted from the [original tweet](https://twitter.com/DasSurma/status/1706086320051794217)

---

---
url: /catalog/c/yoda-condition.md
---
## Rewrite Check to Yoda Condition&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImMiLCJxdWVyeSI6IiRDOiAkVCA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwiY29uZmlnIjoiaWQ6IG1heS10aGUtZm9yY2UtYmUtd2l0aC15b3Vcbmxhbmd1YWdlOiBjXG5ydWxlOlxuICBwYXR0ZXJuOiAkQSA9PSAkQiBcbiAgaW5zaWRlOlxuICAgIGtpbmQ6IHBhcmVudGhlc2l6ZWRfZXhwcmVzc2lvblxuICAgIGluc2lkZToge2tpbmQ6IGlmX3N0YXRlbWVudH1cbmNvbnN0cmFpbnRzOlxuICBCOiB7IGtpbmQ6IG51bWJlcl9saXRlcmFsIH1cbmZpeDogJEIgPT0gJEEiLCJzb3VyY2UiOiJpZiAobXlOdW1iZXIgPT0gNDIpIHsgLyogLi4uICovfVxuaWYgKG5vdE1hdGNoID09IGFub3RoZXIpIHt9XG5pZiAobm90TWF0Y2gpIHt9In0=)

### Description

In programming jargon, a [Yoda condition](https://en.wikipedia.org/wiki/Yoda_conditions) is a style that places the constant portion of the expression on the left side of the conditional statement. It is used to prevent assignment errors that may occur in languages like C.

### YAML

```yaml
id: may-the-force-be-with-you
language: c
rule:
  pattern: $A == $B                 # Find equality comparison
  inside:                           # inside an if_statement
    kind: parenthesized_expression
    inside: {kind: if_statement}
constraints:                        # with the constraint that
  B: { kind: number_literal }       # right side is a number
fix: $B == $A
```

The rule targets an equality comparison, denoted by the [pattern](/guide/pattern-syntax.html) `$A == $B`. This comparison must occur [inside](/reference/rule.html#inside) an `if_statement`. Additionally, there’s a [constraint](/reference/yaml.html#constraints) that the right side of the comparison, `$B`, must be a number\_literal like `42`.

### Example

```c {1}
if (myNumber == 42) { /* ... */}
if (notMatch == another) { /* ... */}
if (notMatch) { /* ... */}
```

### Diff

```c
if (myNumber == 42) { /* ... */} // [!code --]
if (42 == myNumber) { /* ... */} // [!code ++]
if (notMatch == another) { /* ... */}
if (notMatch) { /* ... */}
```

### Contributed by

Inspired by this [thread](https://x.com/cocoa1han/status/1763020689303581141)

---

---
url: /catalog/cpp/find-struct-inheritance.md
---
## Find Struct Inheritance

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiY3BwIiwicXVlcnkiOiJzdHJ1Y3QgJFNPTUVUSElORzogICRJTkhFUklUU19GUk9NIHsgJCQkQk9EWTsgfSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiIsInNvdXJjZSI6InN0cnVjdCBGb286IEJhciB7fTtcblxuc3RydWN0IEJhcjogQmF6IHtcbiAgaW50IGEsIGI7XG59In0=)

### Description

ast-grep's pattern is AST based. A code snippet like `struct $SOMETHING:  $INHERITS` will not work because it does not have a correct AST structure. The correct pattern should spell out the full syntax like `struct $SOMETHING: $INHERITS { $$$BODY; }`.

Compare the ast structure below to see the difference, especially the `ERROR` node. You can also use the playground's pattern panel to debug.

:::code-group

```shell [Wrong Pattern]
ERROR
  $SOMETHING
  base_class_clause
    $INHERITS
```

```shell [Correct Pattern]
struct_specifier
  $SOMETHING
  base_class_clause
    $INHERITS
  field_declaration_list
    field_declaration
      $$$BODY
```

:::

If it is not possible to write a full pattern, [YAML rule](/guide/rule-config.html) is a better choice.

### Pattern

```shell
ast-grep --lang cpp --pattern '
struct $SOMETHING: $INHERITS { $$$BODY; }'
```

### Example

```cpp {1-3}
struct Bar: Baz {
  int a, b;
}
```

### Contributed by

Inspired by this [tweet](https://x.com/techno_bog/status/1885421768384331871)

---

---
url: /catalog/cpp/fix-format-vuln.md
---
## Fix Format String Vulnerability&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImNwcCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IGZpeC1mb3JtYXQtc2VjdXJpdHktZXJyb3Jcbmxhbmd1YWdlOiBDcHBcbnJ1bGU6XG4gIHBhdHRlcm46ICRQUklOVEYoJFMsICRWQVIpXG5jb25zdHJhaW50czpcbiAgUFJJTlRGOiAjIGEgZm9ybWF0IHN0cmluZyBmdW5jdGlvblxuICAgIHsgcmVnZXg6IFwiXnNwcmludGZ8ZnByaW50ZiRcIiB9XG4gIFZBUjogIyBub3QgYSBsaXRlcmFsIHN0cmluZ1xuICAgIG5vdDpcbiAgICAgIGFueTpcbiAgICAgIC0geyBraW5kOiBzdHJpbmdfbGl0ZXJhbCB9XG4gICAgICAtIHsga2luZDogY29uY2F0ZW5hdGVkX3N0cmluZyB9XG5maXg6ICRQUklOVEYoJFMsIFwiJXNcIiwgJFZBUilcbiIsInNvdXJjZSI6Ii8vIEVycm9yXG5mcHJpbnRmKHN0ZGVyciwgb3V0KTtcbnNwcmludGYoJmJ1ZmZlclsyXSwgb2JqLT5UZXh0KTtcbnNwcmludGYoYnVmMSwgVGV4dF9TdHJpbmcoVFhUX1dBSVRJTkdfRk9SX0NPTk5FQ1RJT05TKSk7XG4vLyBPS1xuZnByaW50ZihzdGRlcnIsIFwiJXNcIiwgb3V0KTtcbnNwcmludGYoJmJ1ZmZlclsyXSwgXCIlc1wiLCBvYmotPlRleHQpO1xuc3ByaW50ZihidWYxLCBcIiVzXCIsIFRleHRfU3RyaW5nKFRYVF9XQUlUSU5HX0ZPUl9DT05ORUNUSU9OUykpOyJ9)

### Description

The [Format String exploit](https://owasp.org/www-community/attacks/Format_string_attack) occurs when the submitted data of an input string is evaluated as a command by the application.

For example, using `sprintf(s, var)` can lead to format string vulnerabilities if `var` contains user-controlled data. This can be exploited to execute arbitrary code. By explicitly specifying the format string as `"%s"`, you ensure that `var` is treated as a string, mitigating this risk.

### YAML

```yaml
id: fix-format-security-error
language: Cpp
rule:
  pattern: $PRINTF($S, $VAR)
constraints:
  PRINTF: # a format string function
    { regex: "^sprintf|fprintf$" }
  VAR: # not a literal string
    not:
      any:
      - { kind: string_literal }
      - { kind: concatenated_string }
fix: $PRINTF($S, "%s", $VAR)
```

### Example

```cpp {2-4}
// Error
fprintf(stderr, out);
sprintf(&buffer[2], obj->Text);
sprintf(buf1, Text_String(TXT_WAITING_FOR_CONNECTIONS));
// OK
fprintf(stderr, "%s", out);
sprintf(&buffer[2], "%s", obj->Text);
sprintf(buf1, "%s", Text_String(TXT_WAITING_FOR_CONNECTIONS));
```

### Diff

```js
// Error
fprintf(stderr, out); // [!code --]
fprintf(stderr, "%s", out); // [!code ++]
sprintf(&buffer[2], obj->Text); // [!code --]
sprintf(&buffer[2], "%s", obj->Text); // [!code ++]
sprintf(buf1, Text_String(TXT_WAITING_FOR_CONNECTIONS)); // [!code --]
sprintf(buf1, "%s", Text_String(TXT_WAITING_FOR_CONNECTIONS)); // [!code ++]
// OK
fprintf(stderr, "%s", out);
sprintf(&buffer[2], "%s", obj->Text);
sprintf(buf1, "%s", Text_String(TXT_WAITING_FOR_CONNECTIONS));
```

### Contributed by

[xiaoxiangmoe](https://github.com/xiaoxiangmoe)

---

---
url: /catalog/go/defer-func-call-antipattern.md
---
## Detect problematic defer statements with function calls

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiZ28iLCJxdWVyeSI6InsgXG4gICAgZGVmZXIgJEEuJEIodCwgZmFpbHBvaW50LiRNKCQkJCkpIFxufSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6ImRlZmVyX3N0YXRlbWVudCIsImNvbmZpZyI6InJ1bGU6XG4iLCJzb3VyY2UiOiJmdW5jIFRlc3RJc3N1ZTE2Njk2KHQgKnRlc3RpbmcuVCkge1xuXHRhbGFybVJhdGlvIDo9IHZhcmRlZi5NZW1vcnlVc2FnZUFsYXJtUmF0aW8uTG9hZCgpXG5cdHZhcmRlZi5NZW1vcnlVc2FnZUFsYXJtUmF0aW8uU3RvcmUoMC4wKVxuXHRkZWZlciB2YXJkZWYuTWVtb3J5VXNhZ2VBbGFybVJhdGlvLlN0b3JlKGFsYXJtUmF0aW8pXG5cdHJlcXVpcmUuTm9FcnJvcih0LCBmYWlscG9pbnQuRW5hYmxlKFwiZ2l0aHViLmNvbS9waW5nY2FwL3RpZGIvcGtnL2V4ZWN1dG9yL3NvcnRleGVjL3Rlc3RTb3J0ZWRSb3dDb250YWluZXJTcGlsbFwiLCBcInJldHVybih0cnVlKVwiKSlcblx0ZGVmZXIgcmVxdWlyZS5Ob0Vycm9yKHQsIFxuXHQgICBmYWlscG9pbnQuRGlzYWJsZShcblx0XHRcImdpdGh1Yi5jb20vcGluZ2NhcC90aWRiL3BrZy9leGVjdXRvci9zb3J0ZXhlYy90ZXN0U29ydGVkUm93Q29udGFpbmVyU3BpbGxcIlxuXHQpKVxuXHRyZXF1aXJlLk5vRXJyb3IodCwgXG5cdFx0ZmFpbHBvaW50LkVuYWJsZShcImdpdGh1Yi5jb20vcGluZ2NhcC90aWRiL3BrZy9leGVjdXRvci9qb2luL3Rlc3RSb3dDb250YWluZXJTcGlsbFwiLCBcInJldHVybih0cnVlKVwiKSlcblx0ZGVmZXIgcmVxdWlyZS5Ob0Vycm9yKHQsIFxuXHRcdGZhaWxwb2ludC5EaXNhYmxlKFwiZ2l0aHViLmNvbS9waW5nY2FwL3RpZGIvcGtnL2V4ZWN1dG9yL2pvaW4vdGVzdFJvd0NvbnRhaW5lclNwaWxsXCIpKVxufSJ9)

### Description

This rule detects a common anti-pattern in Go testing code where `defer` statements contain function calls with parameters that are evaluated immediately instead of when the defer executes.

In Go, `defer` schedules a function call to be executed when the surrounding function returns. However, the **arguments to the deferred function are evaluated immediately** when the defer statement is encountered, not when the defer executes.

This is particularly problematic when using assertion libraries in tests. For example:

```go
defer require.NoError(t, failpoint.Disable("some/path"))
```

In this case, `failpoint.Disable("some/path")` is called immediately when the defer statement is reached, not when the function exits. This means the failpoint is disabled right after being enabled, making the test ineffective.

### Pattern

```shell
ast-grep \
  --lang go \
  --pattern '{ defer $A.$B(t, failpoint.$M($$$)) } \
  --selector defer_statement'
```

### Example

```go{6-9,11-12}
func TestIssue16696(t *testing.T) {
	alarmRatio := vardef.MemoryUsageAlarmRatio.Load()
	vardef.MemoryUsageAlarmRatio.Store(0.0)
	defer vardef.MemoryUsageAlarmRatio.Store(alarmRatio)
	require.NoError(t, failpoint.Enable("github.com/pingcap/tidb/pkg/executor/sortexec/testSortedRowContainerSpill", "return(true)"))
	defer require.NoError(t,
	   failpoint.Disable(
		"github.com/pingcap/tidb/pkg/executor/sortexec/testSortedRowContainerSpill"
	))
	require.NoError(t, failpoint.Enable("github.com/pingcap/tidb/pkg/executor/join/testRowContainerSpill", "return(true)"))
	defer require.NoError(t,
		failpoint.Disable("github.com/pingcap/tidb/pkg/executor/join/testRowContainerSpill"))
}
```

### Fix

The correct way to defer a function with parameters is to wrap it in an anonymous function:

```go
defer func() {
    require.NoError(t, failpoint.Disable("some/path"))
}()
```

### Contributed by

Inspired by [YangKeao's tweet](https://x.com/YangKeao/status/1671420857565212672) about this common pitfall in TiDB codebase.

---

---
url: /catalog/go/match-function-call.md
---
## Match Function Call in Golang

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiJhd2FpdCAkQSIsInJld3JpdGUiOiJ0cnkge1xuICAgIGF3YWl0ICRBXG59IGNhdGNoKGUpIHtcbiAgICAvLyB0b2RvXG59IiwiY29uZmlnIjoicnVsZTpcbiAgcGF0dGVybjpcbiAgICBjb250ZXh0OiAnZnVuYyB0KCkgeyBmbXQuUHJpbnRsbigkJCRBKSB9J1xuICAgIHNlbGVjdG9yOiBjYWxsX2V4cHJlc3Npb25cbiIsInNvdXJjZSI6ImZ1bmMgbWFpbigpIHtcbiAgICBmbXQuUHJpbnRsbihcIk9LXCIpXG59In0=)

### Description

One of the common questions of ast-grep is to match function calls in Golang.

A plain pattern like `fmt.Println($A)` will not work. This is because Golang syntax also allows type conversions, e.g. `int(3.14)`, that look like function calls. Tree-sitter, ast-grep's parser, will prefer parsing `func_call(arg)` as a type conversion instead of a call expression.

To avoid this ambiguity, ast-grep lets us write a [contextual pattern](/guide/rule-config/atomic-rule.html#pattern), which is a pattern inside a larger code snippet.
We can use `context` to write a pattern like this: `func t() { fmt.Println($A) }`. Then, we can use the selector `call_expression` to match only function calls.

Please also read the [deep dive](/advanced/pattern-parse.html) on [ambiguous pattern](/advanced/pattern-parse.html#ambiguous-pattern-code).

### YAML

```yaml
id: match-function-call
language: go
rule:
  pattern:
    context: 'func t() { fmt.Println($A) }'
    selector: call_expression
```

### Example

```go{2}
func main() {
    fmt.Println("OK")
}
```

### Contributed by

Inspired by [QuantumGhost](https://github.com/QuantumGhost) from [ast-grep/ast-grep#646](https://github.com/ast-grep/ast-grep/issues/646)

---

---
url: /catalog/go/find-func-declaration-with-prefix.md
---
## Find function declarations with names of certain pattern

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiJyJ15bQS1aYS16MC05Xy1dKyciLCJyZXdyaXRlIjoiIiwiY29uZmlnIjoiaWQ6IHRlc3QtZnVuY3Rpb25zXG5sYW5ndWFnZTogZ29cbnJ1bGU6XG4gIGtpbmQ6IGZ1bmN0aW9uX2RlY2xhcmF0aW9uXG4gIGhhczpcbiAgICBmaWVsZDogbmFtZVxuICAgIHJlZ2V4OiBUZXN0LipcbiIsInNvdXJjZSI6InBhY2thZ2UgYWJzXG5pbXBvcnQgXCJ0ZXN0aW5nXCJcbmZ1bmMgVGVzdEFicyh0ICp0ZXN0aW5nLlQpIHtcbiAgICBnb3QgOj0gQWJzKC0xKVxuICAgIGlmIGdvdCAhPSAxIHtcbiAgICAgICAgdC5FcnJvcmYoXCJBYnMoLTEpID0gJWQ7IHdhbnQgMVwiLCBnb3QpXG4gICAgfVxufVxuIn0=)

### Description

ast-grep can find function declarations by their names. But not all names can be matched by a meta variable pattern. For instance, you cannot use a meta variable pattern to find function declarations whose names start with a specific prefix, e.g. `TestAbs` with the prefix `Test`. Attempting `Test$_` will fail because it is not a valid syntax.

Instead, you can use a [YAML rule](/reference/rule.html) to use the [`regex`](/guide/rule-config/atomic-rule.html#regex) atomic rule.

### YAML

```yaml
id: test-functions
language: go
rule:
  kind: function_declaration
  has:
    field: name
    regex: Test.*
```

### Example

```go{3-8}
package abs
import "testing"
func TestAbs(t *testing.T) {
    got := Abs(-1)
    if got != 1 {
        t.Errorf("Abs(-1) = %d; want 1", got)
    }
}
```

### Contributed by

[kevinkjt2000](https://twitter.com/kevinkjt2000) on [Discord](https://discord.com/invite/4YZjf6htSQ).

---

---
url: /catalog/go/unmarshal-tag-is-dash.md
---
## Detect problematic JSON tags with dash prefix

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiJgJFRBR2AiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogdW5tYXJzaGFsLXRhZy1pcy1kYXNoXG5zZXZlcml0eTogZXJyb3Jcbm1lc3NhZ2U6IFN0cnVjdCBmaWVsZCBjYW4gYmUgZGVjb2RlZCB3aXRoIHRoZSBgLWAga2V5IGJlY2F1c2UgdGhlIEpTT04gdGFnXG4gIHN0YXJ0cyB3aXRoIGEgYC1gIGJ1dCBpcyBmb2xsb3dlZCBieSBhIGNvbW1hLlxucnVsZTpcbiAgcGF0dGVybjogJ2AkVEFHYCdcbiAgaW5zaWRlOlxuICAgIGtpbmQ6IGZpZWxkX2RlY2xhcmF0aW9uXG5jb25zdHJhaW50czpcbiAgVEFHOiBcbiAgICByZWdleDoganNvbjpcIi0sLipcIiIsInNvdXJjZSI6InBhY2thZ2UgbWFpblxuXG50eXBlIFRlc3RTdHJ1Y3QxIHN0cnVjdCB7XG5cdC8vIG9rOiB1bm1hcnNoYWwtdGFnLWlzLWRhc2hcblx0QSBzdHJpbmcgYGpzb246XCJpZFwiYFxufVxuXG50eXBlIFRlc3RTdHJ1Y3QyIHN0cnVjdCB7XG5cdC8vIHJ1bGVpZDogdW5tYXJzaGFsLXRhZy1pcy1kYXNoXG5cdEIgc3RyaW5nIGBqc29uOlwiLSxvbWl0ZW1wdHlcImBcbn1cblxudHlwZSBUZXN0U3RydWN0MyBzdHJ1Y3Qge1xuXHQvLyBydWxlaWQ6IHVubWFyc2hhbC10YWctaXMtZGFzaFxuXHRDIHN0cmluZyBganNvbjpcIi0sMTIzXCJgXG59XG5cbnR5cGUgVGVzdFN0cnVjdDQgc3RydWN0IHtcblx0Ly8gcnVsZWlkOiB1bm1hcnNoYWwtdGFnLWlzLWRhc2hcblx0RCBzdHJpbmcgYGpzb246XCItLFwiYFxufSJ9)

### Description

This rule detects a security vulnerability in Go's JSON unmarshaling. When a struct field has a JSON tag that starts with `-,`, it can be unexpectedly unmarshaled with the `-` key.

According to the [Go documentation](https://pkg.go.dev/encoding/json#Marshal), if the field tag is `-`, the field should be omitted. However, a field with name `-` can still be unmarshaled using the tag `-,`.

This creates a security issue where developers think they are preventing a field from being unmarshaled (like `IsAdmin` in authentication), but attackers can still set that field by providing the `-` key in JSON input.

```go
type User struct {
    Username string `json:"username,omitempty"`
    Password string `json:"password,omitempty"`
    IsAdmin  bool   `json:"-,omitempty"`  // Intended to prevent marshaling
}

// This still works and sets IsAdmin to true!
json.Unmarshal([]byte(`{"-": true}`), &user)
// Result: main.User{Username:"", Password:"", IsAdmin:true}
```

### YAML

```yaml
id: unmarshal-tag-is-dash
severity: error
message: Struct field can be decoded with the `-` key because the JSON tag
  starts with a `-` but is followed by a comma.
rule:
  pattern: '`$TAG`'
  inside:
    kind: field_declaration
constraints:
  TAG:
    regex: json:"-,.*"
```

### Example

```go{8,12,16}
package main

type TestStruct1 struct {
	A string `json:"id"` // ok
}

type TestStruct2 struct {
	B string `json:"-,omitempty"` // wrong
}

type TestStruct3 struct {
	C string `json:"-,123"` // wrong
}

type TestStruct4 struct {
	D string `json:"-,"` // wrong
}
```

### Fix

To properly omit a field from JSON marshaling/unmarshaling, use just `-` without a comma:

```go
type User struct {
    Username string `json:"username,omitempty"`
    Password string `json:"password,omitempty"`
    IsAdmin  bool   `json:"-"`  // Correctly prevents marshaling/unmarshaling
}
```

### Contributed by

Inspired by [Trail of Bits blog post](https://blog.trailofbits.com/2025/06/17/unexpected-security-footguns-in-gos-parsers/) and their [public Semgrep rule](https://semgrep.dev/playground/r/trailofbits.go.unmarshal-tag-is-dash).

---

---
url: /catalog/go/match-package-import.md
---
## Match package import in Golang

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImdvIiwicXVlcnkiOiIiLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogbWF0Y2gtcGFja2FnZS1pbXBvcnRcbmxhbmd1YWdlOiBnb1xucnVsZTpcbiAga2luZDogaW1wb3J0X3NwZWNcbiAgaGFzOlxuICAgIHJlZ2V4OiBnaXRodWIuY29tL2dvbGFuZy1qd3Qvand0Iiwic291cmNlIjoicGFja2FnZSBtYWluXG5cbmltcG9ydCAoXG5cdFwiZm10XCJcblx0XCJnaXRodWIuY29tL2dvbGFuZy1qd3Qvand0XCIgIC8vIFRoaXMgbWF0Y2hlcyB0aGUgQVNUIHJ1bGVcbilcblxuZnVuYyBtYWluKCkge1xuXHQvLyBDcmVhdGUgYSBuZXcgdG9rZW5cblx0dG9rZW4gOj0gand0Lk5ldyhqd3QuU2lnbmluZ01ldGhvZEhTMjU2KVxuXHRcblx0Ly8gQWRkIHNvbWUgY2xhaW1zXG5cdHRva2VuLkNsYWltcyA9IGp3dC5NYXBDbGFpbXN7XG5cdFx0XCJ1c2VyXCI6IFwiYWxpY2VcIixcblx0XHRcInJvbGVcIjogXCJhZG1pblwiLFxuXHR9XG5cdFxuXHQvLyBTaWduIHRoZSB0b2tlblxuXHR0b2tlblN0cmluZywgZXJyIDo9IHRva2VuLlNpZ25lZFN0cmluZyhbXWJ5dGUoXCJteS1zZWNyZXRcIikpXG5cdGlmIGVyciAhPSBuaWwge1xuXHRcdGZtdC5QcmludGYoXCJFcnJvciBzaWduaW5nIHRva2VuOiAldlxcblwiLCBlcnIpXG5cdFx0cmV0dXJuXG5cdH1cblx0XG5cdGZtdC5QcmludGYoXCJHZW5lcmF0ZWQgdG9rZW46ICVzXFxuXCIsIHRva2VuU3RyaW5nKVxufSJ9)

### Description

A generic rule template for detecting imports of specific packages in Go source code. This rule can be customized to match any package by modifying the regex pattern, making it useful for security auditing, dependency management, and compliance checking.

This rule identifies Go import statements based on the configured regex pattern, including:

Direct imports: `import "package/name"`\
Versioned imports: `import "package/name/v4"`\
Subpackage imports: `import "package/name/subpkg"`\
Grouped imports within `import () blocks`

### YAML

```yaml
id: match-package-import
language: go
rule:
  kind: import_spec
  has:
    regex: PACKAGE_PATTERN_HERE
```

### Example

JWT Library Detection

```go{5}
package main

import (
	"fmt"
	"github.com/golang-jwt/jwt" // This matches the AST rule
)

func main() {
	token := jwt.New(jwt.SigningMethodHS256) // Create a new token
	// Add some claims
	token.Claims = jwt.MapClaims{"user": "alice", "role": "admin"}
	tokenString, err := token.SignedString([]byte("my-secret")) // Sign the token
	if err != nil {
		fmt.Printf("Error signing token: %v\n", err)
		return
	}
	fmt.Printf("Generated token: %s\n", tokenString)
}
```

### Contributed by

[Sudesh Gutta](https://github.com/sudeshgutta)

---

---
url: /catalog/html/extract-i18n-key.md
---
## Extract i18n Keys&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6Imh0bWwiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogdGV4dFxuICBwYXR0ZXJuOiAkVFxuICBub3Q6XG4gICAgcmVnZXg6ICdcXHtcXHsuKlxcfVxcfSdcbmZpeDogXCJ7eyAkKCckVCcpIH19XCIiLCJzb3VyY2UiOiI8dGVtcGxhdGU+XG4gIDxzcGFuPkhlbGxvPC9zcGFuPlxuICA8c3Bhbj57eyB0ZXh0IH19PC9zcGFuPlxuPC90ZW1wbGF0ZT4ifQ==)

### Description

It is tedious to manually find and replace all the text in the template with i18n keys. This rule helps to extract static text into i18n keys. Dynamic text, e.g. mustache syntax, will be skipped.

In practice, you may want to map the extracted text to a key in a dictionary file. While this rule only demonstrates the extraction part, further mapping process can be done via a script reading the output of ast-grep's [`--json`](/guide/tools/json.html) mode, or using [`@ast-grep/napi`](/guide/api-usage/js-api.html).

### YAML

```yaml
id: extract-i18n-key
language: html
rule:
  kind: text
  pattern: $T
  # skip dynamic text in mustache syntax
  not: { regex: '\{\{.*\}\}' }
fix: "{{ $('$T') }}"
```

### Example

```html {2}
<template>
  <span>Hello</span>
  <span>{{ text }}</span>
</template>
```

### Diff

```html
<template>
  <span>Hello</span> // [!code --]
  <span>{{ $('Hello') }}</span> // [!code ++]
  <span>{{ text }}</span>
</template>
```

### Contributed by

Inspired by [Vue.js RFC](https://github.com/vuejs/rfcs/discussions/705#discussion-7255672)

---

---
url: /catalog/java/find-field-with-type.md
---
## Find Java field declarations of type String

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmEiLCJxdWVyeSI6ImAkVEFHYCIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIGtpbmQ6IGZpZWxkX2RlY2xhcmF0aW9uXG4gIGhhczpcbiAgICBmaWVsZDogdHlwZVxuICAgIHJlZ2V4OiBeU3RyaW5nJCIsInNvdXJjZSI6IkBDb21wb25lbnRcbmNsYXNzIEFCQyBleHRlbmRzIE9iamVjdHtcbiAgICBAUmVzb3VyY2VcbiAgICBwcml2YXRlIGZpbmFsIFN0cmluZyB3aXRoX2Fubm87XG5cbiAgICBwcml2YXRlIGZpbmFsIFN0cmluZyB3aXRoX211bHRpX21vZDtcblxuICAgIHB1YmxpYyBTdHJpbmcgc2ltcGxlO1xufSJ9)

### Description

To extract all Java field names of type `String` is not as straightforward as one might think. A simple pattern like `String $F;` would only match fields declared without any modifiers or annotations. However, a pattern like `$MOD String $F;` cannot be correctly parsed by tree-sitter.

:::details Use playground pattern debugger to explore the AST

You can use the [playground](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YSIsInF1ZXJ5IjoiY2xhc3MgQUJDe1xuICAgJE1PRCBTdHJpbmcgdGVzdDtcbn0iLCJyZXdyaXRlIjoiIiwic3RyaWN0bmVzcyI6ImFzdCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTpcbiAga2luZDogZmllbGRfZGVjbGFyYXRpb25cbiAgaGFzOlxuICAgIGZpZWxkOiB0eXBlXG4gICAgcmVnZXg6IF5TdHJpbmckIiwic291cmNlIjoiQENvbXBvbmVudFxuY2xhc3MgQUJDIGV4dGVuZHMgT2JqZWN0e1xuICAgIEBSZXNvdXJjZVxuICAgIHByaXZhdGUgZmluYWwgU3RyaW5nIHdpdGhfYW5ubztcblxuICAgIHByaXZhdGUgZmluYWwgU3RyaW5nIHdpdGhfbXVsdGlfbW9kO1xuXG4gICAgcHVibGljIFN0cmluZyBzaW1wbGU7XG59In0=)'s pattern tab to visualize the AST of `class A { $MOD String $F; }`.

```
field_declaration
  $MOD
  variable_declarator
    identifier: String
  ERROR
    identifier: $F
```

Tree-sitter does not think `$MOD` is a valid modifier, so it produces an `ERROR`.

While the valid AST for code like `private String field;` produces different AST structures:

```
field_declaration
  modifiers
  type_identifier
  variable_declarator
    identifier: field
```

:::

A more robust approach is to use a structural rule that targets `field_declaration` nodes and applies a `has` constraint on the `type` child node to match the type `String`. This method effectively captures fields regardless of their modifiers or annotations.

### YAML

```yaml
id: find-field-with-type
language: java
rule:
  kind: field_declaration
  has:
    field: type
    regex: ^String$
```

### Example

```java {3-4,6,8}
@Component
class ABC extends Object{
    @Resource
    private final String with_anno;

    private final String with_multi_mod;

    public String simple;
}
```

### Contributed by

Inspired by the post [discussion](https://github.com/ast-grep/ast-grep/discussions/2195)

---

---
url: /catalog/java/no-unused-vars.md
---
## No Unused Vars in Java&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmEiLCJxdWVyeSI6ImlmKHRydWUpeyQkJEJPRFl9IiwicmV3cml0ZSI6IiRDOiBMaXN0WyRUXSA9IHJlbGF0aW9uc2hpcCgkJCRBLCB1c2VsaXN0PVRydWUsICQkJEIpIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogbm8tdW51c2VkLXZhcnNcbnJ1bGU6XG4gICAga2luZDogbG9jYWxfdmFyaWFibGVfZGVjbGFyYXRpb25cbiAgICBhbGw6XG4gICAgICAgIC0gaGFzOlxuICAgICAgICAgICAgaGFzOlxuICAgICAgICAgICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgICAgICAgICBwYXR0ZXJuOiAkSURFTlRcbiAgICAgICAgLSBub3Q6XG4gICAgICAgICAgICBwcmVjZWRlczpcbiAgICAgICAgICAgICAgICBzdG9wQnk6IGVuZFxuICAgICAgICAgICAgICAgIGhhczpcbiAgICAgICAgICAgICAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAgICAgICAgICAgICAgICAgYW55OlxuICAgICAgICAgICAgICAgICAgICAgICAgLSB7IGtpbmQ6IGlkZW50aWZpZXIsIHBhdHRlcm46ICRJREVOVCB9XG4gICAgICAgICAgICAgICAgICAgICAgICAtIHsgaGFzOiB7a2luZDogaWRlbnRpZmllciwgcGF0dGVybjogJElERU5ULCBzdG9wQnk6IGVuZH19XG5maXg6ICcnXG4iLCJzb3VyY2UiOiJTdHJpbmcgdW51c2VkID0gXCJ1bnVzZWRcIjtcbk1hcDxTdHJpbmcsIFN0cmluZz4gZGVjbGFyZWRCdXROb3RJbnN0YW50aWF0ZWQ7XG5cblN0cmluZyB1c2VkMSA9IFwidXNlZFwiO1xuaW50IHVzZWQyID0gMztcbmJvb2xlYW4gdXNlZDMgPSBmYWxzZTtcbmludCB1c2VkNCA9IDQ7XG5TdHJpbmcgdXNlZDUgPSBcIjVcIjtcblxuXG5cbnVzZWQxO1xuU3lzdGVtLm91dC5wcmludGxuKHVzZWQyKTtcbmlmKHVzZWQzKXtcbiAgICBTeXN0ZW0ub3V0LnByaW50bG4oXCJzb21lIHZhcnMgYXJlIHVudXNlZFwiKTtcbiAgICBNYXA8U3RyaW5nLCBTdHJpbmc+IHVudXNlZE1hcCA9IG5ldyBIYXNoTWFwPD4oKSB7e1xuICAgICAgICBwdXQodXNlZDUsIFwidXNlZDVcIik7XG4gICAgfX07XG5cbiAgICAvLyBFdmVuIHRob3VnaCB3ZSBkb24ndCByZWFsbHkgZG8gYW55dGhpbmcgd2l0aCB0aGlzIG1hcCwgc2VwYXJhdGluZyB0aGUgZGVjbGFyYXRpb24gYW5kIGluc3RhbnRpYXRpb24gbWFrZXMgaXQgY291bnQgYXMgYmVpbmcgdXNlZFxuICAgIGRlY2xhcmVkQnV0Tm90SW5zdGFudGlhdGVkID0gbmV3IEhhc2hNYXA8PigpO1xuXG4gICAgcmV0dXJuIHVzZWQ0O1xufSJ9)

### Description

Identifying unused variables is a common task in code refactoring. You should rely on a Java linter or IDE for this task rather than writing a custom rule in ast-grep, but for educational purposes, this rule demonstrates how to find unused variables in Java.

This approach makes some simplifying assumptions. We only consider local variable declarations and ignore the other many ways variables can be declared: Method Parameters, Fields, Class Variables, Constructor Parameters, Loop Variables, Exception Handler Parameters, Lambda Parameters, Annotation Parameters, Enum Constants, and Record Components. Now you may see why it is recommended to use a rule from an established linter or IDE rather than writing your own.

### YAML

```yaml
id: no-unused-vars
rule:
    kind: local_variable_declaration
    all:
        - has:
            has:
                kind: identifier
                pattern: $IDENT
        - not:
            precedes:
                stopBy: end
                has:
                    stopBy: end
                    any:
                        - { kind: identifier, pattern: $IDENT }
                        - { has: {kind: identifier, pattern: $IDENT, stopBy: end}}
fix: ''
```

First, we identify the local variable declaration and capture the pattern of the identifier inside of it. Then we use `not` and `precedes` to only match the local variable declaration if the identifier we captured does not appear later in the code.

It is important to note that we use `all` here to force the ordering of the `has` rule to be before the `not` rule. This guarantees that the meta-variable `$IDENT` is captured by looking inside of the local variable declaration.

Additionally, when looking ahead in the code, we can't just look for the identifier directly, but for any node that may contain the identifier.

### Example

```java
String unused = "unused"; // [!code --]
String used = "used";
System.out.println(used);
```

---

---
url: /catalog/kotlin/ensure-clean-architecture.md
---
## Ensure Clean Architecture

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImtvdGxpbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogaW1wb3J0LWRlcGVuZGVuY3ktdmlvbGF0aW9uXG5tZXNzYWdlOiBJbXBvcnQgRGVwZW5kZW5jeSBWaW9sYXRpb24gXG5ub3RlczogRW5zdXJlcyB0aGF0IGltcG9ydHMgY29tcGx5IHdpdGggYXJjaGl0ZWN0dXJhbCBydWxlcy4gXG5zZXZlcml0eTogZXJyb3JcbnJ1bGU6XG4gIHBhdHRlcm46IGltcG9ydCAkUEFUSFxuY29uc3RyYWludHM6XG4gIFBBVEg6XG4gICAgYW55OlxuICAgIC0gcmVnZXg6IGNvbVxcLmV4YW1wbGUoXFwuXFx3KykqXFwuZGF0YVxuICAgIC0gcmVnZXg6IGNvbVxcLmV4YW1wbGUoXFwuXFx3KykqXFwucHJlc2VudGF0aW9uXG5maWxlczpcbi0gY29tL2V4YW1wbGUvZG9tYWluLyoqLyoua3QiLCJzb3VyY2UiOiJpbXBvcnQgYW5kcm9pZHgubGlmZWN5Y2xlLlZpZXdNb2RlbFxuaW1wb3J0IGFuZHJvaWR4LmxpZmVjeWNsZS5WaWV3TW9kZWxTY29wZVxuaW1wb3J0IGNvbS5leGFtcGxlLmN1c3RvbWxpbnRleGFtcGxlLmRhdGEubW9kZWxzLlVzZXJEdG9cbmltcG9ydCBjb20uZXhhbXBsZS5jdXN0b21saW50ZXhhbXBsZS5kb21haW4udXNlY2FzZXMuR2V0VXNlclVzZUNhc2VcbmltcG9ydCBjb20uZXhhbXBsZS5jdXN0b21saW50ZXhhbXBsZS5wcmVzZW50YXRpb24uc3RhdGVzLk1haW5TdGF0ZVxuaW1wb3J0IGRhZ2dlci5oaWx0LmFuZHJvaWQubGlmZWN5Y2xlLkhpbHRWaWV3TW9kZWwifQ==)

### Description

This ast-grep rule ensures that the **domain** package in a [Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) project does not import classes from the **data** or **presentation** packages. It enforces the separation of concerns by preventing the domain layer from depending on other layers, maintaining the integrity of the architecture.

For example, the rule will trigger an error if an import statement like `import com.example.data.SomeClass` or `import com.example.presentation.AnotherClass` is found within the domain package.

The rule uses the [`files`](/reference/yaml.html#files) field to apply only to the domain package.

### YAML

```yaml
id: import-dependency-violation
message: Import Dependency Violation
notes: Ensures that imports comply with architectural rules.
severity: error
rule:
  pattern: import $PATH  # capture the import statement
constraints:
  PATH: # find specific package imports
    any:
    - regex: com\.example(\.\w+)*\.data
    - regex: com\.example(\.\w+)*\.presentation
files:  # apply only to domain package
- com/example/domain/**/*.kt
```

### Example

```kotlin {3,5}
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelScope
import com.example.customlintexample.data.models.UserDto
import com.example.customlintexample.domain.usecases.GetUserUseCase
import com.example.customlintexample.presentation.states.MainState
import dagger.hilt.android.lifecycle.HiltViewModel
```

### Contributed by

Inspired by the post [Custom Lint Task Configuration in Gradle with Kotlin DSL](https://www.sngular.com/insights/320/custom-lint-task-configuration-in-gradle-with-kotlin-dsl)

---

---
url: /catalog/python/migrate-openai-sdk.md
---
## Migrate OpenAI SDK&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmICRGVU5DKCQkJEFSR1MpOiAkJCRCT0RZIiwicmV3cml0ZSI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IGltcG9ydCBvcGVuYWlcbmZpeDogZnJvbSBvcGVuYWkgaW1wb3J0IENsaWVudFxuLS0tXG5ydWxlOlxuICBwYXR0ZXJuOiBvcGVuYWkuYXBpX2tleSA9ICRLRVlcbmZpeDogY2xpZW50ID0gQ2xpZW50KCRLRVkpXG4tLS1cbnJ1bGU6XG4gIHBhdHRlcm46IG9wZW5haS5Db21wbGV0aW9uLmNyZWF0ZSgkJCRBUkdTKVxuZml4OiB8LVxuICBjbGllbnQuY29tcGxldGlvbnMuY3JlYXRlKFxuICAgICQkJEFSR1NcbiAgKSIsInNvdXJjZSI6ImltcG9ydCBvc1xuaW1wb3J0IG9wZW5haVxuZnJvbSBmbGFzayBpbXBvcnQgRmxhc2ssIGpzb25pZnlcblxuYXBwID0gRmxhc2soX19uYW1lX18pXG5vcGVuYWkuYXBpX2tleSA9IG9zLmdldGVudihcIk9QRU5BSV9BUElfS0VZXCIpXG5cblxuQGFwcC5yb3V0ZShcIi9jaGF0XCIsIG1ldGhvZHM9KFwiUE9TVFwiKSlcbmRlZiBpbmRleCgpOlxuICAgIGFuaW1hbCA9IHJlcXVlc3QuZm9ybVtcImFuaW1hbFwiXVxuICAgIHJlc3BvbnNlID0gb3BlbmFpLkNvbXBsZXRpb24uY3JlYXRlKFxuICAgICAgICBtb2RlbD1cInRleHQtZGF2aW5jaS0wMDNcIixcbiAgICAgICAgcHJvbXB0PWdlbmVyYXRlX3Byb21wdChhbmltYWwpLFxuICAgICAgICB0ZW1wZXJhdHVyZT0wLjYsXG4gICAgKVxuICAgIHJldHVybiBqc29uaWZ5KHJlc3BvbnNlLmNob2ljZXMpIn0=)

### Description

OpenAI has introduced some breaking changes in their API, such as using `Client` to initialize the service and renaming the `Completion` method to `completions` . This example shows how to use ast-grep to automatically update your code to the new API.

API migration requires multiple related rules to work together.
The example shows how to write [multiple rules](/reference/playground.html#test-multiple-rules) in a [single YAML](/guide/rewrite-code.html#using-fix-in-yaml-rule) file.
The rules and patterns in the example are simple and self-explanatory, so we will not explain them further.

### YAML

```yaml
id: import-openai
language: python
rule:
  pattern: import openai
fix: from openai import Client
---
id: rewrite-client
language: python
rule:
  pattern: openai.api_key = $KEY
fix: client = Client($KEY)
---
id: rewrite-chat-completion
language: python
rule:
  pattern: openai.Completion.create($$$ARGS)
fix: |-
  client.completions.create(
    $$$ARGS
  )
```

### Example

```python {2,6,11-15}
import os
import openai
from flask import Flask, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

@app.route("/chat", methods=("POST"))
def index():
    animal = request.form["animal"]
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=generate_prompt(animal),
        temperature=0.6,
    )
    return jsonify(response.choices)
```

### Diff

```python
import os
import openai # [!code --]
from openai import Client # [!code ++]
from flask import Flask, jsonify

app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY") # [!code --]
client = Client(os.getenv("OPENAI_API_KEY")) # [!code ++]

@app.route("/chat", methods=("POST"))
def index():
    animal = request.form["animal"]
    response = openai.Completion.create( # [!code --]
    response = client.completions.create( # [!code ++]
      model="text-davinci-003",
      prompt=generate_prompt(animal),
      temperature=0.6,
    )
    return jsonify(response.choices)
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [Morgante](https://twitter.com/morgantepell/status/1721668781246750952) from [grit.io](https://www.grit.io/)

---

---
url: /catalog/python/prefer-generator-expressions.md
---
## Prefer Generator Expressions&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiWyQkJEFdIiwicmV3cml0ZSI6IiRBPy4oKSIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46ICRGVU5DKCRMSVNUKVxuY29uc3RyYWludHM6XG4gIExJU1Q6IHsga2luZDogbGlzdF9jb21wcmVoZW5zaW9uIH1cbiAgRlVOQzpcbiAgICBhbnk6XG4gICAgICAtIHBhdHRlcm46IGFueVxuICAgICAgLSBwYXR0ZXJuOiBhbGxcbiAgICAgIC0gcGF0dGVybjogc3VtXG4gICAgICAjIC4uLlxudHJhbnNmb3JtOlxuICBJTk5FUjpcbiAgICBzdWJzdHJpbmc6IHtzb3VyY2U6ICRMSVNULCBzdGFydENoYXI6IDEsIGVuZENoYXI6IC0xIH1cbmZpeDogJEZVTkMoJElOTkVSKSIsInNvdXJjZSI6ImFsbChbeCBmb3IgeCBpbiB5XSlcblt4IGZvciB4IGluIHldIn0=)

### Description

List comprehensions like `[x for x in range(10)]` are a concise way to create lists in Python. However, we can achieve better memory efficiency by using generator expressions like `(x for x in range(10))` instead. List comprehensions create the entire list in memory, while generator expressions generate each element one at a time. We can make the change by replacing the square brackets with parentheses.

### YAML

```yaml
id: prefer-generator-expressions
language: python
rule:
  pattern: $LIST
  kind: list_comprehension
transform:
  INNER:
    substring: {source: $LIST, startChar: 1, endChar: -1 }
fix: ($INNER)
```

This rule converts every list comprehension to a generator expression. However, **not every list comprehension can be replaced with a generator expression.** If the list is used multiple times, is modified, is sliced, or is indexed, a generator is not a suitable replacement.

Some common functions like `any`, `all`, and `sum` take an `iterable` as an argument. A generator function counts as an `iterable`, so it is safe to change a list comprehension to a generator expression in this context.

```yaml
id: prefer-generator-expressions
language: python
rule:
  pattern: $FUNC($LIST)
constraints:
  LIST: { kind: list_comprehension }
  FUNC:
    any:
      - pattern: any
      - pattern: all
      - pattern: sum
      # ...
transform:
  INNER:
    substring: {source: $LIST, startChar: 1, endChar: -1 }
fix: $FUNC($INNER)
```

### Example

```python
any([x for x in range(10)])
```

### Diff

```python
any([x for x in range(10)]) # [!code --]
any(x for x in range(10)) # [!code ++]
```

### Contributed by

[Steven Love](https://github.com/StevenLove)

---

---
url: /catalog/python/optional-to-none-union.md
---
## Rewrite `Optional[Type]` to `Type | None`&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJzaWduYXR1cmUiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIHBhdHRlcm46IFxuICAgIGNvbnRleHQ6ICdhOiBPcHRpb25hbFskVF0nXG4gICAgc2VsZWN0b3I6IGdlbmVyaWNfdHlwZVxuZml4OiAkVCB8IE5vbmUiLCJzb3VyY2UiOiJkZWYgYShhcmc6IE9wdGlvbmFsW0ludF0pOiBwYXNzIn0=)

### Description

[PEP 604](https://peps.python.org/pep-0604/) recommends that `Type | None` is preferred over `Optional[Type]` for Python 3.10+.

This rule performs such rewriting. Note `Optional[$T]` alone is interpreted as subscripting expression instead of generic type, we need to use [pattern object](/guide/rule-config/atomic-rule.html#pattern-object) to disambiguate it with more context code.

### YAML

```yaml
id: optional-to-none-union
language: python
rule:
  pattern:
    context: 'a: Optional[$T]'
    selector: generic_type
fix: $T | None
```

### Example

```py {1}
def a(arg: Optional[int]): pass
```

### Diff

```py
def a(arg: Optional[int]): pass # [!code --]
def a(arg: int | None): pass # [!code ++]
```

### Contributed by

[Bede Carroll](https://github.com/ast-grep/ast-grep/discussions/1492)

---

---
url: /catalog/python/refactor-pytest-fixtures.md
---
## Refactor pytest fixtures

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZGVmIGZvbygkWCk6XG4gICRTIiwicmV3cml0ZSI6ImxvZ2dlci5sb2coJE1BVENIKSIsImNvbmZpZyI6ImlkOiBweXRlc3QtdHlwZS1oaW50LWZpeHR1cmVcbmxhbmd1YWdlOiBQeXRob25cbnV0aWxzOlxuICBpcy1maXh0dXJlLWZ1bmN0aW9uOlxuICAgIGtpbmQ6IGZ1bmN0aW9uX2RlZmluaXRpb25cbiAgICBmb2xsb3dzOlxuICAgICAga2luZDogZGVjb3JhdG9yXG4gICAgICBoYXM6XG4gICAgICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgICAgcmVnZXg6IF5maXh0dXJlJFxuICAgICAgICBzdG9wQnk6IGVuZFxuICBpcy10ZXN0LWZ1bmN0aW9uOlxuICAgIGtpbmQ6IGZ1bmN0aW9uX2RlZmluaXRpb25cbiAgICBoYXM6XG4gICAgICBmaWVsZDogbmFtZVxuICAgICAgcmVnZXg6IF50ZXN0X1xuICBpcy1weXRlc3QtY29udGV4dDpcbiAgICAjIFB5dGVzdCBjb250ZXh0IGlzIGEgbm9kZSBpbnNpZGUgYSBweXRlc3RcbiAgICAjIHRlc3QvZml4dHVyZVxuICAgIGluc2lkZTpcbiAgICAgIHN0b3BCeTogZW5kXG4gICAgICBhbnk6XG4gICAgICAgIC0gbWF0Y2hlczogaXMtZml4dHVyZS1mdW5jdGlvblxuICAgICAgICAtIG1hdGNoZXM6IGlzLXRlc3QtZnVuY3Rpb25cbiAgaXMtZml4dHVyZS1hcmc6XG4gICAgIyBGaXh0dXJlIGFyZ3VtZW50cyBhcmUgaWRlbnRpZmllcnMgaW5zaWRlIHRoZSBcbiAgICAjIHBhcmFtZXRlcnMgb2YgYSB0ZXN0L2ZpeHR1cmUgZnVuY3Rpb25cbiAgICBhbGw6XG4gICAgICAtIGtpbmQ6IGlkZW50aWZpZXJcbiAgICAgIC0gbWF0Y2hlczogaXMtcHl0ZXN0LWNvbnRleHRcbiAgICAgIC0gaW5zaWRlOlxuICAgICAgICAgIGtpbmQ6IHBhcmFtZXRlcnNcbnJ1bGU6XG4gIG1hdGNoZXM6IGlzLWZpeHR1cmUtYXJnXG4gIHJlZ2V4OiBeZm9vJFxuZml4OiAnZm9vOiBpbnQnXG4iLCJzb3VyY2UiOiJmcm9tIGNvbGxlY3Rpb25zLmFiYyBpbXBvcnQgSXRlcmFibGVcbmZyb20gdHlwaW5nIGltcG9ydCBBbnlcblxuaW1wb3J0IHB5dGVzdFxuZnJvbSBweXRlc3QgaW1wb3J0IGZpeHR1cmVcblxuQHB5dGVzdC5maXh0dXJlKHNjb3BlPVwic2Vzc2lvblwiKVxuZGVmIGZvbygpIC0+IEl0ZXJhYmxlW2ludF06XG4gICAgeWllbGQgNVxuXG5AZml4dHVyZVxuZGVmIGJhcihmb28pIC0+IHN0cjpcbiAgICByZXR1cm4gc3RyKGZvbylcblxuZGVmIHJlZ3VsYXJfZnVuY3Rpb24oZm9vKSAtPiBOb25lOlxuICAgICMgVGhpcyBmdW5jdGlvbiBkb2Vzbid0IHVzZSB0aGUgJ2ZvbycgZml4dHVyZVxuICAgIHByaW50KGZvbylcblxuZGVmIHRlc3RfMShmb28sIGJhcik6XG4gICAgcHJpbnQoZm9vLCBiYXIpXG5cbmRlZiB0ZXN0XzIoYmFyKTpcbiAgICAuLi4ifQ==)

### Description

One of the most commonly used testing framework in Python is [pytest](https://docs.pytest.org/en/8.2.x/). Among other things, it allows the use of [fixtures](https://docs.pytest.org/en/6.2.x/fixture.html).

Fixtures are defined as functions that can be required in test code, or in other fixtures, as an argument. This means that all functions arguments with a given name in a pytest context (test function or fixture) are essentially the same entity. However, not every editor's LSP is able to keep track of this, making refactoring challenging.

Using ast-grep, we can define some rules to match fixture definition and usage without catching similarly named entities in a non-test context.

First, we define utils to select pytest test/fixture functions.

```yaml
utils:
  is-fixture-function:
    kind: function_definition
    follows:
      kind: decorator
      has:
        kind: identifier
        regex: ^fixture$
        stopBy: end
  is-test-function:
    kind: function_definition
    has:
      field: name
      regex: ^test_
```

Pytest fixtures are declared with a decorator `@pytest.fixture`. We match the `function_definition` node that directly follows a `decorator` node. That decorator node must have a `fixture` identifier somewhere. This accounts for different location of the `fixture` node depending on the type of imports and whether the decorator is used as is or called with parameters.

Pytest functions are fairly straightforward to detect, as they always start with `test_` by convention.

The next utils builds onto those two to incrementally:

* Find if a node is inside a pytest context (test/fixture)
* Find if a node is an argument in such a context

```yaml
utils:
  is-pytest-context:
    # Pytest context is a node inside a pytest
    # test/fixture
    inside:
      stopBy: end
      any:
        - matches: is-fixture-function
        - matches: is-test-function
  is-fixture-arg:
    # Fixture arguments are identifiers inside the 
    # parameters of a test/fixture function
    all:
      - kind: identifier
      - inside:
          kind: parameters
      - matches: is-pytest-context
```

Once those utils are declared, you can perform various refactoring on a specific fixture.

The following rule adds a type-hint to a fixture.

```yaml
rule:
  matches: is-fixture-arg
  regex: ^foo$
fix: 'foo: int'
```

This one renames a fixture and all its references.

```yaml
rule:
  kind: identifier
  matches: is-fixture-context
  regex: ^foo$
fix: 'five'
```

### Example

#### Renaming Fixtures

```python {2,6,7,12,13}
@pytest.fixture
def foo() -> int:
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo: int) -> str:
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo: int) -> None:
    assert foo == 5
```

#### Diff

```python {2,6,7,12}
@pytest.fixture
def foo() -> int: # [!code --]
def five() -> int: # [!code ++]
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo: int) -> str: # [!code --]
def some_fixture(five: int) -> str: # [!code ++]
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo: int) -> None: # [!code --]
def test_code(five: int) -> None: # [!code ++]
    assert foo == 5 # [!code --]
    assert five == 5 # [!code ++]
```

#### Type Hinting Fixtures

```python {6,12}
@pytest.fixture
def foo() -> int:
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo) -> str:
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo) -> None:
    assert foo == 5
```

#### Diff

```python {2,6,7,12}
@pytest.fixture
def foo() -> int:
    return 5

@pytest.fixture(scope="function")
def some_fixture(foo) -> str: # [!code --]
def some_fixture(foo: int) -> str: # [!code ++]
    return str(foo)

def regular_function(foo) -> None:
    ...

def test_code(foo) -> None: # [!code --]
def test_code(foo: int) -> None: # [!code ++]
    assert foo == 5
```

---

---
url: /catalog/python/use-walrus-operator-in-if.md
---
## Use Walrus Operator in `if` statement

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InB5dGhvbiIsInF1ZXJ5IjoiZm4gbWFpbigpIHsgXG4gICAgJCQkO1xuICAgIGlmKCRBKXskJCRCfSBcbiAgICBpZigkQSl7JCQkQ30gXG4gICAgJCQkRlxufSIsInJld3JpdGUiOiJmbiBtYWluKCkgeyAkJCRFOyBpZigkQSl7JCQkQiAkJCRDfSAkJCRGfSIsImNvbmZpZyI6ImlkOiB1c2Utd2FscnVzLW9wZXJhdG9yXG5ydWxlOlxuICBmb2xsb3dzOlxuICAgIHBhdHRlcm46XG4gICAgICBjb250ZXh0OiAkVkFSID0gJCQkRVhQUlxuICAgICAgc2VsZWN0b3I6IGV4cHJlc3Npb25fc3RhdGVtZW50XG4gIHBhdHRlcm46IFwiaWYgJFZBUjogJCQkQlwiXG5maXg6IHwtXG4gIGlmICRWQVIgOj0gJCQkRVhQUjpcbiAgICAkJCRCXG4tLS1cbmlkOiByZW1vdmUtZGVjbGFyYXRpb25cbnJ1bGU6XG4gIHBhdHRlcm46XG4gICAgY29udGV4dDogJFZBUiA9ICQkJEVYUFJcbiAgICBzZWxlY3RvcjogZXhwcmVzc2lvbl9zdGF0ZW1lbnRcbiAgcHJlY2VkZXM6XG4gICAgcGF0dGVybjogXCJpZiAkVkFSOiAkJCRCXCJcbmZpeDogJyciLCJzb3VyY2UiOiJhID0gZm9vKClcblxuaWYgYTpcbiAgICBkb19iYXIoKSJ9)

### Description

The walrus operator (`:=`) introduced in Python 3.8 allows you to assign values to variables as part of an expression. This rule aims to simplify code by using the walrus operator in `if` statements.

This first part of the rule identifies cases where a variable is assigned a value and then immediately used in an `if` statement to control flow.

```yaml
id: use-walrus-operator
language: python
rule:
  pattern: "if $VAR: $$$B"
  follows:
    pattern:
      context: $VAR = $$$EXPR
      selector: expression_statement
fix: |-
  if $VAR := $$$EXPR:
    $$$B
```

The `pattern` clause finds an `if` statement that checks the truthiness of `$VAR`.
If this pattern `follows` an expression statement where `$VAR` is assigned `$$$EXPR`, the `fix` clause changes the `if` statements to use the walrus operator.

The second part of the rule:

```yaml
id: remove-declaration
rule:
  pattern:
    context: $VAR = $$$EXPR
    selector: expression_statement
  precedes:
    pattern: "if $VAR: $$$B"
fix: ''
```

This rule removes the standalone variable assignment when it directly precedes an `if` statement that uses the walrus operator. Since the assignment is now part of the `if` statement, the separate declaration is no longer needed.

By applying these rules, you can refactor your Python code to be more concise and readable, taking advantage of the walrus operator's ability to combine an assignment with an expression.

### YAML

```yaml
id: use-walrus-operator
language: python
rule:
  follows:
    pattern:
      context: $VAR = $$$EXPR
      selector: expression_statement
  pattern: "if $VAR: $$$B"
fix: |-
  if $VAR := $$$EXPR:
    $$$B
---
id: remove-declaration
language: python
rule:
  pattern:
    context: $VAR = $$$EXPR
    selector: expression_statement
  precedes:
    pattern: "if $VAR: $$$B"
fix: ''
```

### Example

```python
a = foo()

if a:
    do_bar()
```

### Diff

```python
a = foo() # [!code --]

if a: # [!code --]
if a := foo(): # [!code ++]
    do_bar()
```

### Contributed by

Inspired by reddit user [/u/jackerhack](https://www.reddit.com/r/rust/comments/13eg738/comment/kagdklw/?)

---

---
url: /catalog/ruby/migrate-action-filter.md
---
## Migrate action\_filter in Ruby on Rails&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1YnkiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiIyBhc3QtZ3JlcCBZQU1MIFJ1bGUgaXMgcG93ZXJmdWwgZm9yIGxpbnRpbmchXG4jIGh0dHBzOi8vYXN0LWdyZXAuZ2l0aHViLmlvL2d1aWRlL3J1bGUtY29uZmlnLmh0bWwjcnVsZVxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogYmVmb3JlX2ZpbHRlciAkJCRBQ1RJT05cbiAgICAtIHBhdHRlcm46IGFyb3VuZF9maWx0ZXIgJCQkQUNUSU9OXG4gICAgLSBwYXR0ZXJuOiBhZnRlcl9maWx0ZXIgJCQkQUNUSU9OXG4gIGhhczpcbiAgICBwYXR0ZXJuOiAkRklMVEVSXG4gICAgZmllbGQ6IG1ldGhvZFxuZml4OiBcbiAgJE5FV19BQ1RJT04gJCQkQUNUSU9OXG50cmFuc2Zvcm06XG4gIE5FV19BQ1RJT046XG4gICAgcmVwbGFjZTpcbiAgICAgIHNvdXJjZTogJEZJTFRFUlxuICAgICAgcmVwbGFjZTogX2ZpbHRlclxuICAgICAgYnk6IF9hY3Rpb24iLCJzb3VyY2UiOiJjbGFzcyBUb2Rvc0NvbnRyb2xsZXIgPCBBcHBsaWNhdGlvbkNvbnRyb2xsZXJcbiAgYmVmb3JlX2ZpbHRlciA6YXV0aGVudGljYXRlXG4gIGFyb3VuZF9maWx0ZXIgOndyYXBfaW5fdHJhbnNhY3Rpb24sIG9ubHk6IDpzaG93XG4gIGFmdGVyX2ZpbHRlciBkbyB8Y29udHJvbGxlcnxcbiAgICBmbGFzaFs6ZXJyb3JdID0gXCJZb3UgbXVzdCBiZSBsb2dnZWQgaW5cIlxuICBlbmRcblxuICBkZWYgaW5kZXhcbiAgICBAdG9kb3MgPSBUb2RvLmFsbFxuICBlbmRcbmVuZFxuIn0=)

### Description

This rule is used to migrate `{before,after,around}_filter` to `{before,after,around}_action` in Ruby on Rails controllers.

These are methods that run before, after or around an action is executed, and they can be used to check permissions, set variables, redirect requests, log events, etc. However, these methods are [deprecated](https://stackoverflow.com/questions/16519828/rails-4-before-filter-vs-before-action) in Rails 5.0 and will be removed in Rails 5.1. `{before,after,around}_action` are the new syntax for the same functionality.

This rule will replace all occurrences of `{before,after,around}_filter` with `{before,after,around}_action` in the controller code.

### YAML

```yaml
id: migration-action-filter
language: ruby
rule:
  any:
    - pattern: before_filter $$$ACTION
    - pattern: around_filter $$$ACTION
    - pattern: after_filter $$$ACTION
  has:
    pattern: $FILTER
    field: method
fix:
  $NEW_ACTION $$$ACTION
transform:
  NEW_ACTION:
    replace:
      source: $FILTER
      replace: _filter
      by: _action
```

### Example

```rb {2-4}
class TodosController < ApplicationController
  before_filter :authenticate
  around_filter :wrap_in_transaction, only: :show
  after_filter do |controller|
    flash[:error] = "You must be logged in"
  end

  def index
    @todos = Todo.all
  end
end
```

### Diff

```rb
class TodosController < ApplicationController
  before_action :authenticate  # [!code --]
  before_filter :authenticate # [!code ++]
  around_action :wrap_in_transaction, only: :show # [!code --]
  around_filter :wrap_in_transaction, only: :show # [!code ++]
  after_action do |controller|  # [!code --]
     flash[:error] = "You must be logged in" # [!code --]
  end # [!code --]
  after_filter do |controller| # [!code ++]
    flash[:error] = "You must be logged in" # [!code ++]
  end # [!code ++]

  def index
    @todos = Todo.all
  end
end
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [Future of Ruby - AST Tooling](https://dev.to/baweaver/future-of-ruby-ast-tooling-9i1).

---

---
url: /catalog/ruby/prefer-symbol-over-proc.md
---
## Prefer Symbol over Proc&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1YnkiLCJxdWVyeSI6IiRMSVNULnNlbGVjdCB7IHwkVnwgJFYuJE1FVEhPRCB9IiwicmV3cml0ZSI6IiRMSVNULnNlbGVjdCgmOiRNRVRIT0QpIiwiY29uZmlnIjoiaWQ6IHByZWZlci1zeW1ib2wtb3Zlci1wcm9jXG5ydWxlOlxuICBwYXR0ZXJuOiAkTElTVC4kSVRFUiB7IHwkVnwgJFYuJE1FVEhPRCB9XG5sYW5ndWFnZTogUnVieVxuY29uc3RyYWludHM6XG4gIElURVI6XG4gICAgcmVnZXg6ICdtYXB8c2VsZWN0fGVhY2gnXG5maXg6ICckTElTVC4kSVRFUigmOiRNRVRIT0QpJ1xuIiwic291cmNlIjoiWzEsIDIsIDNdLnNlbGVjdCB7IHx2fCB2LmV2ZW4/IH1cbigxLi4xMDApLmVhY2ggeyB8aXwgaS50b19zIH1cbm5vdF9saXN0Lm5vX21hdGNoIHsgfHZ8IHYuZXZlbj8gfVxuIn0=)

### Description

Ruby has a more concise symbol shorthand `&:` to invoke methods.
This rule simplifies `proc` to `symbol`.
This example is inspired by this [dev.to article](https://dev.to/baweaver/future-of-ruby-ast-tooling-9i1).

### YAML

```yaml
id: prefer-symbol-over-proc
language: ruby
rule:
  pattern: $LIST.$ITER { |$V| $V.$METHOD }
constraints:
  ITER:
    regex: 'map|select|each'
fix: '$LIST.$ITER(&:$METHOD)'
```

### Example

```rb {1,2}
[1, 2, 3].select { |v| v.even? }
(1..100).each { |i| i.to_s }
not_list.no_match { |v| v.even? }
```

### Diff

```rb
[1, 2, 3].select { |v| v.even? } # [!code --]
[1, 2, 3].select(&:even?) # [!code ++]
(1..100).each { |i| i.to_s } # [!code --]
(1..100).each(&:to_s) # [!code ++]

not_list.no_match { |v| v.even? }
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim)

---

---
url: /catalog/rust/avoid-duplicated-exports.md
---
## Avoid Duplicated Exports

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IiIsImNvbmZpZyI6InJ1bGU6XG4gIGFsbDpcbiAgICAgLSBwYXR0ZXJuOiBwdWIgdXNlICRCOjokQztcbiAgICAgLSBpbnNpZGU6XG4gICAgICAgIGtpbmQ6IHNvdXJjZV9maWxlXG4gICAgICAgIGhhczpcbiAgICAgICAgICBwYXR0ZXJuOiBwdWIgbW9kICRBO1xuICAgICAtIGhhczpcbiAgICAgICAgcGF0dGVybjogJEFcbiAgICAgICAgc3RvcEJ5OiBlbmQiLCJzb3VyY2UiOiJwdWIgbW9kIGZvbztcbnB1YiB1c2UgZm9vOjpGb287XG5wdWIgdXNlIGZvbzo6QTo6QjtcblxuXG5wdWIgdXNlIGFhYTo6QTtcbnB1YiB1c2Ugd29vOjpXb287In0=)

### Description

Generally, we don't encourage the use of re-exports.

However, sometimes, to keep the interface exposed by a lib crate tidy, we use re-exports to shorten the path to specific items.
When doing so, a pitfall is to export a single item under two different names.

Consider:

```rs
pub mod foo;
pub use foo::Foo;
```

The issue with this code, is that `Foo` is now exposed under two different paths: `Foo`, `foo::Foo`.

This unnecessarily increases the surface of your API.
It can also cause issues on the client side. For example, it makes the usage of auto-complete in the IDE more involved.

Instead, ensure you export only once with `pub`.

### YAML

```yaml
id: avoid-duplicate-export
language: rust
rule:
  all:
     - pattern: pub use $B::$C;
     - inside:
        kind: source_file
        has:
          pattern: pub mod $A;
     - has:
        pattern: $A
        stopBy: end
```

### Example

```rs {2,3}
pub mod foo;
pub use foo::Foo;
pub use foo::A::B;


pub use aaa::A;
pub use woo::Woo;
```

### Contributed by

Julius Lungys([voidpumpkin](https://github.com/voidpumpkin))

---

---
url: /catalog/rust/boshen-footgun.md
---
## Beware of char offset when iterate over a string&#x20;

* [Playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoicnVzdCIsInF1ZXJ5IjoiJEEuY2hhcnMoKS5lbnVtZXJhdGUoKSIsInJld3JpdGUiOiIkQS5jaGFyX2luZGljZXMoKSIsImNvbmZpZyI6IiIsInNvdXJjZSI6ImZvciAoaSwgY2hhcikgaW4gc291cmNlLmNoYXJzKCkuZW51bWVyYXRlKCkge1xuICAgIHByaW50bG4hKFwiQm9zaGVuIGlzIGFuZ3J5IDopXCIpO1xufSJ9)

### Description

It's a common pitfall in Rust that counting *character offset* is not the same as counting *byte offset* when iterating through a string. Rust string is represented by utf-8 byte array, which is a variable-length encoding scheme.

`chars().enumerate()` will yield the character offset, while [`char_indices()`](https://doc.rust-lang.org/std/primitive.str.html#method.char_indices) will yield the byte offset.

```rs
let yes = "y̆es";
let mut char_indices = yes.char_indices();
assert_eq!(Some((0, 'y')), char_indices.next()); // not (0, 'y̆')
assert_eq!(Some((1, '\u{0306}')), char_indices.next());
// note the 3 here - the last character took up two bytes
assert_eq!(Some((3, 'e')), char_indices.next());
assert_eq!(Some((4, 's')), char_indices.next());
```

Depending on your use case, you may want to use `char_indices()` instead of `chars().enumerate()`.

### Pattern

```shell
ast-grep -p '$A.chars().enumerate()' \
   -r '$A.char_indices()' \
   -l rs
```

### Example

```rs {1}
for (i, char) in source.chars().enumerate() {
    println!("Boshen is angry :)");
}
```

### Diff

```rs
for (i, char) in source.chars().enumerate() { // [!code --]
for (i, char) in source.char_indices() { // [!code ++]
    println!("Boshen is angry :)");
}
```

### Contributed by

Inspired by [Boshen's Tweet](https://x.com/boshen_c/status/1719033308682870891)

![Boshen's footgun](https://pbs.twimg.com/media/F9s7mJHaYAEndnY?format=jpg\&name=medium)

---

---
url: /catalog/rust/redundant-unsafe-function.md
---
## Unsafe Function Without Unsafe Block

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IntcbiAgZGVzY3JpcHRpb24gPSAkQVxufSIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6ImJpbmRpbmciLCJjb25maWciOiIgIGlkOiByZWR1bmRhbnQtdW5zYWZlLWZ1bmN0aW9uXG4gIGxhbmd1YWdlOiBydXN0XG4gIHNldmVyaXR5OiBlcnJvclxuICBtZXNzYWdlOiBVbnNhZmUgZnVuY3Rpb24gd2l0aG91dCB1bnNhZmUgYmxvY2sgaW5zaWRlXG4gIG5vdGU6IHxcbiAgICBDb25zaWRlciB3aGV0aGVyIHRoaXMgZnVuY3Rpb24gbmVlZHMgdG8gYmUgbWFya2VkIHVuc2FmZSBcbiAgICBvciBpZiB1bnNhZmUgb3BlcmF0aW9ucyBzaG91bGQgYmUgd3JhcHBlZCBpbiBhbiB1bnNhZmUgYmxvY2tcbiAgcnVsZTpcbiAgICBhbGw6XG4gICAgICAtIGtpbmQ6IGZ1bmN0aW9uX2l0ZW1cbiAgICAgIC0gaGFzOlxuICAgICAgICAgIGtpbmQ6IGZ1bmN0aW9uX21vZGlmaWVyc1xuICAgICAgICAgIHJlZ2V4OiBcIl51bnNhZmVcIlxuICAgICAgLSBub3Q6XG4gICAgICAgICAgaGFzOlxuICAgICAgICAgICAga2luZDogdW5zYWZlX2Jsb2NrXG4gICAgICAgICAgICBzdG9wQnk6IGVuZCIsInNvdXJjZSI6IiAgLy8gU2hvdWxkIG1hdGNoIC0gdW5zYWZlIGZ1bmN0aW9uIHdpdGhvdXQgdW5zYWZlIGJsb2NrIChubyByZXR1cm4gdHlwZSlcbiAgdW5zYWZlIGZuIHJlZHVuZGFudF91bnNhZmUoKSB7XG4gICAgICBwcmludGxuIShcIk5vIHVuc2FmZSBvcGVyYXRpb25zIGhlcmVcIik7XG4gIH1cblxuICAvLyBTaG91bGQgbWF0Y2ggLSB1bnNhZmUgZnVuY3Rpb24gd2l0aCByZXR1cm4gdHlwZSwgbm8gdW5zYWZlIGJsb2NrXG4gIHVuc2FmZSBmbiByZWR1bmRhbnRfd2l0aF9yZXR1cm4oKSAtPiBpMzIge1xuICAgICAgbGV0IHggPSA1O1xuICAgICAgeCArIDEwXG4gIH1cblxuICAvLyBTaG91bGQgbWF0Y2ggLSB1bnNhZmUgZnVuY3Rpb24gd2l0aCBjb21wbGV4IHJldHVybiB0eXBlXG4gIHVuc2FmZSBmbiByZWR1bmRhbnRfY29tcGxleF9yZXR1cm4oKSAtPiBSZXN1bHQ8U3RyaW5nLCBzdGQ6OmlvOjpFcnJvcj4ge1xuICAgICAgT2soU3RyaW5nOjpmcm9tKFwic2FmZSBvcGVyYXRpb25cIikpXG4gIH1cblxuICAvLyBTaG91bGQgTk9UIG1hdGNoIC0gdW5zYWZlIGZ1bmN0aW9uIHdpdGggdW5zYWZlIGJsb2NrXG4gIHVuc2FmZSBmbiBwcm9wZXJfdW5zYWZlKCkgLT4gKmNvbnN0IGkzMiB7XG4gICAgICB1bnNhZmUge1xuICAgICAgICAgIGxldCBwdHIgPSAweDEyMzQgYXMgKmNvbnN0IGkzMjtcbiAgICAgICAgICBwdHJcbiAgICAgIH1cbiAgfVxuXG4gIC8vIFNob3VsZCBtYXRjaCAtIHVuc2FmZSBhc3luYyBmdW5jdGlvbiB3aXRob3V0IHVuc2FmZSBibG9ja1xuICB1bnNhZmUgYXN5bmMgZm4gYXN5bmNfcmVkdW5kYW50KCkgLT4gaTMyIHtcbiAgICAgIDQyXG4gIH1cblxuICAvLyBTaG91bGQgbWF0Y2ggLSB1bnNhZmUgY29uc3QgZnVuY3Rpb25cbiAgdW5zYWZlIGNvbnN0IGZuIGNvbnN0X3JlZHVuZGFudCgpIC0+IGkzMiB7XG4gICAgICAxMDBcbiAgfVxuXG4gIC8vIFNob3VsZCBOT1QgbWF0Y2ggLSByZWd1bGFyIGZ1bmN0aW9uXG4gIGZuIHJlZ3VsYXJfZnVuY3Rpb24oKSAtPiBpMzIge1xuICAgICAgNDJcbiAgfSJ9)

### Description

This rule detects functions marked with the `unsafe` keyword that do not contain any `unsafe` blocks in their body.

When a function is marked `unsafe`, it indicates that the function contains operations that the compiler cannot verify as safe. However, if the function body doesn't contain any `unsafe` blocks, it may be unnecessarily marked as `unsafe`. This could be a sign that:

1. The function should not be marked `unsafe` if it doesn't perform any unsafe operations
2. Or if there are unsafe operations, they should be explicitly wrapped in `unsafe` blocks for clarity

This rule helps identify such cases so developers can review whether the `unsafe` marker is truly necessary or if the code needs to be refactored.

### YAML

```yaml
id: redundant-unsafe-function
language: rust
severity: error
message: Unsafe function without unsafe block inside
note: |
  Consider whether this function needs to be marked unsafe 
  or if unsafe operations should be wrapped in an unsafe block
rule:
  all:
    - kind: function_item
    - has:
        kind: function_modifiers
        regex: "^unsafe"
    - not:
        has:
          kind: unsafe_block
          stopBy: end
```

### Example

```rs {2,7,12,24,29}
// Should match - unsafe function without unsafe block (no return type)
unsafe fn redundant_unsafe() {
    println!("No unsafe operations here");
}

// Should match - unsafe function with return type, no unsafe block
unsafe fn redundant_with_return() -> i32 {
    let x = 5;
    x + 10
}

// Should match - unsafe function with complex return type
unsafe fn redundant_complex_return() -> Result<String, std::io::Error> {
    Ok(String::from("safe operation"))
}

// Should NOT match - unsafe function with unsafe block
unsafe fn proper_unsafe() -> *const i32 {
    unsafe {
        let ptr = 0x1234 as *const i32;
        ptr
    }
}

// Should match - unsafe async function without unsafe block
unsafe async fn async_redundant() -> i32 {
    42
}

// Should match - unsafe const function
unsafe const fn const_redundant() -> i32 {
    100
}

// Should NOT match - regular function
fn regular_function() -> i32 {
    42
}
```

### Contributed by

Inspired by [@hd\_nvim's Tweet](https://x.com/hd_nvim/status/1992810384072585397?s=20)

---

---
url: /catalog/rust/get-digit-count-in-usize.md
---
## Get number of digits in a `usize`&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoicnVzdCIsInF1ZXJ5IjoiJE5VTS50b19zdHJpbmcoKS5jaGFycygpLmNvdW50KCkiLCJyZXdyaXRlIjoiJE5VTS5jaGVja2VkX2lsb2cxMCgpLnVud3JhcF9vcigwKSArIDEiLCJjb25maWciOiIjIFlBTUwgUnVsZSBpcyBtb3JlIHBvd2VyZnVsIVxuIyBodHRwczovL2FzdC1ncmVwLmdpdGh1Yi5pby9ndWlkZS9ydWxlLWNvbmZpZy5odG1sI3J1bGVcbnJ1bGU6XG4gIGFueTpcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUubG9nKCRBKVxuICAgIC0gcGF0dGVybjogY29uc29sZS5kZWJ1ZygkQSlcbmZpeDpcbiAgbG9nZ2VyLmxvZygkQSkiLCJzb3VyY2UiOiJsZXQgd2lkdGggPSAobGluZXMgKyBudW0pLnRvX3N0cmluZygpLmNoYXJzKCkuY291bnQoKTsifQ==)

### Description

Getting the number of digits in a usize number can be useful for various purposes, such as counting the column width of line numbers in a text editor or formatting the output of a number with commas or spaces.

A common but inefficient way of getting the number of digits in a `usize` number is to use `num.to_string().chars().count()`. This method converts the number to a string, iterates over its characters, and counts them. However, this method involves allocating a new string, which can be costly in terms of memory and time.

A better alternative is to use [`checked_ilog10`](https://doc.rust-lang.org/std/primitive.usize.html#method.checked_ilog10).

```rs
num.checked_ilog10().unwrap_or(0) + 1
```

The snippet above computes the integer logarithm base 10 of the number and adds one. This snippet does not allocate any memory and is faster than the string conversion approach. The [efficient](https://doc.rust-lang.org/src/core/num/int_log10.rs.html) `checked_ilog10` function returns an `Option<usize>` that is `Some(log)` if the number is positive and `None` if the number is zero. The `unwrap_or(0)` function returns the value inside the option or `0` if the option is `None`.

### Pattern

```shell
ast-grep -p '$NUM.to_string().chars().count()' \
   -r '$NUM.checked_ilog10().unwrap_or(0) + 1' \
   -l rs
```

### Example

```rs {1}
let width = (lines + num).to_string().chars().count();
```

### Diff

```rs
let width = (lines + num).to_string().chars().count(); // [!code --]
let width = (lines + num).checked_ilog10().unwrap_or(0) + 1; // [!code ++]
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [dogfooding ast-grep](https://github.com/ast-grep/ast-grep/issues/550)

---

---
url: /catalog/rust/rewrite-indoc-macro.md
---
## Rewrite `indoc!` macro&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoicnVzdCIsInF1ZXJ5IjoiaW5kb2MhIHsgciNcIiQkJEFcIiMgfSIsInJld3JpdGUiOiJgJCQkQWAiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTogXG4gYW55OlxuIC0gcGF0dGVybjogJFYgPT09ICRTRU5TRVRJVkVXT1JEXG4gLSBwYXR0ZXJuOiAkU0VOU0VUSVZFV09SRCA9PT0gJFZcbmNvbnN0cmFpbnRzOlxuICBTRU5TRVRJVkVXT1JEOlxuICAgIHJlZ2V4OiBwYXNzd29yZCIsInNvdXJjZSI6ImZuIG1haW4oKSB7XG4gICAgaW5kb2MhIHtyI1wiXG4gICAgICAgIC5mb28ge1xuICAgICAgICAgICAgb3JkZXI6IDE7XG4gICAgICAgIH1cbiAgICBcIiN9O1xufSJ9)

### Description

This example, created from [a Tweet](https://x.com/zack_overflow/status/1885065128590401551), shows a refactoring operation being performed on Rust source code. The changes involve removing `indoc!` macro declarations while preserving the CSS-like content within them.

Previously, the same refactor is implemented by a *unreadable monster regex* in vim syntax.

:::details Click to see the original regex (neovim, btw)

```vimscript
:%s/\v(indoc!|)(| )([|\{)r#"(([^#]+|\n+)+)"#/`\4`
```

I have to confess that I don't understand this regex even if I use neovim, btw.

Let Claude break it down piece by piece:

* `:%s/` - Vim substitution command for all lines
* `\v` - Very magic mode in vim for simpler regex syntax
* `(indoc!|)` - First capture group: matches either "indoc!" or nothing
* `(| )` - Second capture group: matches either empty string or a space
* `([|\{)` - Third capture group: matches either `[` or `{`
* `r#"` - Matches literal `r#"` (Rust raw string delimiter)
* `(([^#]+|\n+)+)` - Fourth capture group (nested):
  * `[^#]+` - One or more non-# characters
  * `|\n+` - OR one or more newlines
  * Outer `()+` makes this repeat one or more times
* `"#` - Matches the closing raw string delimiter
* \`\4\` - Replaces with the fourth capture group wrapped in backticks

This regex is designed to find Rust raw string literals (possibly wrapped in `indoc!` macro), capture their content, and replace the entire match with just the content wrapped in backticks. It's more precise than my previous explanation and matches the pattern you're showing.

:::

### Pattern

```shell
ast-grep --pattern 'indoc! { r#"$$$A"# }' --rewrite '`$$$A`' sgtest.rs
```

### Example

```rs {2-6}
fn main() {
    indoc! {r#"
        .foo {
            order: 1;
        }
    "#};
}
```

### Diff

```rs
fn main() {
    indoc! {r#" // [!code --]
    `.foo {    // [!code ++]
        order: 1;
    }
    "#}; // [!code --]
        `; // [!code ++]
}
```

### Contributed by

[Zack in SF](https://x.com/zack_overflow)

---

---
url: /catalog/rust/rust-2024-let-chain-candidate.md
---
## Rust 2024 Let-Chain Candidate

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6ImlkOiBydXN0LTIwMjQtbGV0LWNoYWluLWNhbmRpZGF0ZVxubGFuZ3VhZ2U6IFJ1c3RcbnNldmVyaXR5OiBoaW50XG5tZXNzYWdlOiBuZXN0ZWQgaWYgY29uZGl0aW9ucyBjYW4gYmUgY29sbGFwc2VkIGludG8gYSBSdXN0IDIwMjQgbGV0LWNoYWluXG5cbnV0aWxzOlxuICAjIFRoZSBuZXN0ZWQgaWYgc3RhdGVtZW50IHNob3VsZCBiZSB0aGUgb25seSBzdGF0ZW1lbnQgaW4gaXRzIGJsb2NrLlxuICBzb2xlLWNoaWxkOlxuICAgIGFsbDpcbiAgICAgIC0gbnRoQ2hpbGQ6IDFcbiAgICAgIC0gbnRoQ2hpbGQ6XG4gICAgICAgICAgcG9zaXRpb246IDFcbiAgICAgICAgICByZXZlcnNlOiB0cnVlXG5cbiAgIyBMZXQtY2hhaW5zIGNhbiBvbmx5IHByZXNlcnZlIGJlaGF2aW9yIHdoZW4gdGhlcmUgaXMgbm8gZWxzZSBicmFuY2guXG4gIGlmLW5vLWVsc2U6XG4gICAga2luZDogaWZfZXhwcmVzc2lvblxuICAgIG5vdDpcbiAgICAgIGhhczpcbiAgICAgICAgZmllbGQ6IGFsdGVybmF0aXZlXG4gICAgICAgIGtpbmQ6IGVsc2VfY2xhdXNlXG5cbiAgIyBBdCBsZWFzdCBvbmUgc2lkZSBvZiB0aGUgbmVzdGVkIGlmIHBhaXIgbXVzdCBiZSBhbiBpZiBsZXQuXG4gIGlmLWxldC1uby1lbHNlOlxuICAgIG1hdGNoZXM6IGlmLW5vLWVsc2VcbiAgICBoYXM6XG4gICAgICBmaWVsZDogY29uZGl0aW9uXG4gICAgICBraW5kOiBsZXRfY29uZGl0aW9uXG5cbiAgIyBNYXRjaCBhbiBpbm5lciBpZiB0aGF0IGlzIHRoZSBzb2xlIHN0YXRlbWVudCBpbnNpZGUgdGhlIG91dGVyIGlmIGJsb2NrLlxuICBzb2xlLWlubmVyLWlmLXN0bXQ6XG4gICAga2luZDogZXhwcmVzc2lvbl9zdGF0ZW1lbnRcbiAgICBtYXRjaGVzOiBzb2xlLWNoaWxkXG4gICAgaGFzOlxuICAgICAgbWF0Y2hlczogaWYtbm8tZWxzZVxuXG4gICMgU2FtZSBhcyBhYm92ZSwgYnV0IHJlcXVpcmUgdGhlIGlubmVyIGlmIHRvIGJlIGFuIGlmIGxldC5cbiAgc29sZS1pbm5lci1pZi1sZXQtc3RtdDpcbiAgICBraW5kOiBleHByZXNzaW9uX3N0YXRlbWVudFxuICAgIG1hdGNoZXM6IHNvbGUtY2hpbGRcbiAgICBoYXM6XG4gICAgICBtYXRjaGVzOiBpZi1sZXQtbm8tZWxzZVxuXG5ydWxlOlxuICAjIFN0YXJ0IGZyb20gYW4gb3V0ZXIgaWYgd2l0aG91dCBhbiBlbHNlIGJyYW5jaC5cbiAgbWF0Y2hlczogaWYtbm8tZWxzZVxuICAjIEl0cyBkaXJlY3QgY29uc2VxdWVuY2UgYmxvY2sgbXVzdCBjb250YWluIGV4YWN0bHkgb25lIGlubmVyIGlmIHN0YXRlbWVudC5cbiAgaGFzOlxuICAgIGZpZWxkOiBjb25zZXF1ZW5jZVxuICAgIGtpbmQ6IGJsb2NrXG4gICAgaGFzOlxuICAgICAgbWF0Y2hlczogc29sZS1pbm5lci1pZi1zdG10XG4gICMgRWl0aGVyIHRoZSBvdXRlciBjb25kaXRpb24gb3IgdGhlIGlubmVyIGNvbmRpdGlvbiBtdXN0IGJpbmQgd2l0aCBpZiBsZXQuXG4gIGFueTpcbiAgICAtIG1hdGNoZXM6IGlmLWxldC1uby1lbHNlXG4gICAgLSBoYXM6XG4gICAgICAgIGZpZWxkOiBjb25zZXF1ZW5jZVxuICAgICAgICBraW5kOiBibG9ja1xuICAgICAgICBoYXM6XG4gICAgICAgICAgbWF0Y2hlczogc29sZS1pbm5lci1pZi1sZXQtc3RtdCIsInNvdXJjZSI6ImZuIGhhbmRsZV9yZXF1ZXN0KHVzZXI6IE9wdGlvbjxVc2VyPiwgY2ZnOiBDb25maWcpIHtcbiAgICBpZiBsZXQgU29tZSh1c2VyKSA9IHVzZXIge1xuICAgICAgICBpZiB1c2VyLmlzX2FjdGl2ZSgpIHtcbiAgICAgICAgICAgIGdyYW50X2FjY2Vzcyh1c2VyKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIGlmIGNmZyEoZGVidWdfYXNzZXJ0aW9ucykge1xuICAgICAgICBpZiBsZXQgU29tZShwYXRoKSA9IGNmZy5sb2dfcGF0aCgpIHtcbiAgICAgICAgICAgIGVuYWJsZV9maWxlX2xvZ2dpbmcocGF0aCk7XG4gICAgICAgIH1cbiAgICB9XG5cbiAgICAvLyBPSzogdGhlIGlubmVyIGlmIGlzIG5vdCB0aGUgb25seSBzdGF0ZW1lbnQgaW4gdGhlIGJsb2NrLlxuICAgIGlmIGxldCBTb21lKHVzZXIpID0gY3VycmVudF91c2VyKCkge1xuICAgICAgICBhdWRpdCgmdXNlcik7XG4gICAgICAgIGlmIHVzZXIuaXNfYWRtaW4oKSB7XG4gICAgICAgICAgICBncmFudF9hZG1pbih1c2VyKTtcbiAgICAgICAgfVxuICAgIH1cblxuICAgIC8vIE9LOiB0aGlzIGhhcyBhbiBlbHNlIGJyYW5jaC5cbiAgICBpZiBsZXQgU29tZShwYXRoKSA9IGNmZy5jYWNoZV9wYXRoKCkge1xuICAgICAgICBpZiBwYXRoLmV4aXN0cygpIHtcbiAgICAgICAgICAgIGxvYWRfY2FjaGUocGF0aCk7XG4gICAgICAgIH1cbiAgICB9IGVsc2Uge1xuICAgICAgICByZWJ1aWxkX2NhY2hlKCk7XG4gICAgfVxufSJ9)

### Description

Rust 2024 supports [`let` conditions in boolean chains](https://doc.rust-lang.org/edition-guide/rust-2024/let-chains.html), so an `if let` wrapping a single nested `if` can often be collapsed into one condition.

This rule reports the outer `if` only when neither branch has an `else`, the outer block contains a single inner `if` statement, and either the outer or inner condition is an `if let`.

### YAML

```yaml
id: rust-2024-let-chain-candidate
language: Rust
severity: hint
message: nested if conditions can be collapsed into a Rust 2024 let-chain

utils:
  # The nested if statement should be the only statement in its block.
  sole-child:
    all:
      - nthChild: 1
      - nthChild:
          position: 1
          reverse: true

  # Let-chains can only preserve behavior when there is no else branch.
  if-no-else:
    kind: if_expression
    not:
      has:
        field: alternative
        kind: else_clause

  # At least one side of the nested if pair must be an if let.
  if-let-no-else:
    matches: if-no-else
    has:
      field: condition
      kind: let_condition

  # Match an inner if that is the sole statement inside the outer if block.
  sole-inner-if-stmt:
    kind: expression_statement
    matches: sole-child
    has:
      matches: if-no-else

  # Same as above, but require the inner if to be an if let.
  sole-inner-if-let-stmt:
    kind: expression_statement
    matches: sole-child
    has:
      matches: if-let-no-else

rule:
  # Start from an outer if without an else branch.
  matches: if-no-else
  # Its direct consequence block must contain exactly one inner if statement.
  has:
    field: consequence
    kind: block
    has:
      matches: sole-inner-if-stmt
  # Either the outer condition or the inner condition must bind with if let.
  any:
    - matches: if-let-no-else
    - has:
        field: consequence
        kind: block
        has:
          matches: sole-inner-if-let-stmt
```

### Example

```rs {2,8}
fn handle_request(user: Option<User>, cfg: Config) {
    if let Some(user) = user {
        if user.is_active() {
            grant_access(user);
        }
    }

    if cfg!(debug_assertions) {
        if let Some(path) = cfg.log_path() {
            enable_file_logging(path);
        }
    }

    // OK: the inner if is not the only statement in the block.
    if let Some(user) = current_user() {
        audit(&user);
        if user.is_admin() {
            grant_admin(user);
        }
    }

    // OK: this has an else branch.
    if let Some(path) = cfg.cache_path() {
        if path.exists() {
            load_cache(path);
        }
    } else {
        rebuild_cache();
    }
}
```

### Contributed by

[Codex](https://openai.com/codex/) (GPT-5.5 high fast)

---

---
url: /catalog/tsx/avoid-jsx-short-circuit.md
---
## Avoid `&&` short circuit in JSX&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiY29uc29sZS5sb2coJE1BVENIKSIsInJld3JpdGUiOiJsb2dnZXIubG9nKCRNQVRDSCkiLCJjb25maWciOiJpZDogZG8td2hhdC1icm9vb29vb2tseW4tc2FpZFxubGFuZ3VhZ2U6IFRzeFxuc2V2ZXJpdHk6IGVycm9yXG5ydWxlOlxuICBraW5kOiBqc3hfZXhwcmVzc2lvblxuICBoYXM6XG4gICAgcGF0dGVybjogJEEgJiYgJEJcbiAgbm90OlxuICAgIGluc2lkZTpcbiAgICAgIGtpbmQ6IGpzeF9hdHRyaWJ1dGVcbmZpeDogXCJ7JEEgPyAkQiA6IG51bGx9XCIiLCJzb3VyY2UiOiI8ZGl2PntcbiAgbnVtICYmIDxkaXYvPlxufTwvZGl2PiJ9)

### Description

In [React](https://react.dev/learn/conditional-rendering), you can conditionally render JSX using JavaScript syntax like `if` statements, `&&`, and `? :` operators.
However, you should almost never put numbers on the left side of `&&`. This is because React will render the number `0`, instead of the JSX element on the right side. A concrete example will be conditionally rendering a list when the list is not empty.

This rule will find and fix any short-circuit rendering in JSX and rewrite it to a ternary operator.

### YAML

```yaml
id: do-what-brooooooklyn-said
language: Tsx
rule:
  kind: jsx_expression
  has:
    pattern: $A && $B
  not:
    inside:
      kind: jsx_attribute
fix: "{$A ? $B : null}"
```

### Example

```tsx {1}
<div>{ list.length && list.map(i => <p/>) }</div>
```

### Diff

```tsx
<div>{ list.length && list.map(i => <p/>) }</div> // [!code --]
<div>{ list.length ?  list.map(i => <p/>) : null }</div> // [!code ++]
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim), inspired by [@Brooooook\_lyn](https://twitter.com/Brooooook_lyn/status/1666637274757595141)

---

---
url: /catalog/tsx/avoid-nested-links.md
---
## Avoid nested links

* [Playground Link](https://ast-grep.github.io/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiaWYgKCRBKSB7ICQkJEIgfSIsInJld3JpdGUiOiJpZiAoISgkQSkpIHtcbiAgICByZXR1cm47XG59XG4kJCRCIiwic3RyaWN0bmVzcyI6InNtYXJ0Iiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogbm8tbmVzdGVkLWxpbmtzXG5sYW5ndWFnZTogdHN4XG5zZXZlcml0eTogZXJyb3JcbnJ1bGU6XG4gIHBhdHRlcm46IDxhICQkJD4kJCRBPC9hPlxuICBoYXM6XG4gICAgcGF0dGVybjogPGEgJCQkPiQkJDwvYT5cbiAgICBzdG9wQnk6IGVuZCIsInNvdXJjZSI6ImZ1bmN0aW9uIENvbXBvbmVudCgpIHtcbiAgcmV0dXJuIDxhIGhyZWY9Jy9kZXN0aW5hdGlvbic+XG4gICAgPGEgaHJlZj0nL2Fub3RoZXJkZXN0aW5hdGlvbic+TmVzdGVkIGxpbmshPC9hPlxuICA8L2E+O1xufVxuZnVuY3Rpb24gT2theUNvbXBvbmVudCgpIHtcbiAgcmV0dXJuIDxhIGhyZWY9Jy9kZXN0aW5hdGlvbic+XG4gICAgSSBhbSBqdXN0IGEgbGluay5cbiAgPC9hPjtcbn0ifQ==)

### Description

React will produce a warning message if you nest a link element inside of another link element. This rule will catch this mistake!

### YAML

```yaml
id: no-nested-links
language: tsx
severity: error
rule:
  pattern: <a $$$>$$$A</a>
  has:
    pattern: <a $$$>$$$</a>
    stopBy: end
```

### Example

```tsx {1-5}
function Component() {
  return <a href='/destination'>
    <a href='/anotherdestination'>Nested link!</a>
  </a>;
}
function OkayComponent() {
  return <a href='/destination'>
    I am just a link.
  </a>;
}
```

### Contributed by

[Tom MacWright](https://macwright.com/)

---

---
url: /catalog/tsx/redundant-usestate-type.md
---
## Unnecessary `useState` Type&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoidHlwZXNjcmlwdCIsInF1ZXJ5IjoidXNlU3RhdGU8c3RyaW5nPigkQSkiLCJyZXdyaXRlIjoidXNlU3RhdGUoJEEpIiwiY29uZmlnIjoiIyBZQU1MIFJ1bGUgaXMgbW9yZSBwb3dlcmZ1bCFcbiMgaHR0cHM6Ly9hc3QtZ3JlcC5naXRodWIuaW8vZ3VpZGUvcnVsZS1jb25maWcuaHRtbCNydWxlXG5ydWxlOlxuICBhbnk6XG4gICAgLSBwYXR0ZXJuOiBjb25zb2xlLmxvZygkQSlcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUuZGVidWcoJEEpXG5maXg6XG4gIGxvZ2dlci5sb2coJEEpIiwic291cmNlIjoiZnVuY3Rpb24gQ29tcG9uZW50KCkge1xuICBjb25zdCBbbmFtZSwgc2V0TmFtZV0gPSB1c2VTdGF0ZTxzdHJpbmc+KCdSZWFjdCcpXG59In0=)

### Description

React's [`useState`](https://react.dev/reference/react/useState) is a Hook that lets you add a state variable to your component. The type annotation of `useState`'s generic type argument, for example `useState<number>(123)`, is unnecessary if TypeScript can infer the type of the state variable from the initial value.

We can usually skip annotating if the generic type argument is a single primitive type like `number`, `string` or `boolean`.

### Pattern

::: code-group

```bash [number]
ast-grep -p 'useState<number>($A)' -r 'useState($A)' -l tsx
```

```bash [string]
ast-grep -p 'useState<string>($A)' -r 'useState($A)'
```

```bash [boolean]
ast-grep -p 'useState<boolean>($A)' -r 'useState($A)'
```

:::

### Example

```ts {2}
function Component() {
  const [name, setName] = useState<string>('React')
}
```

### Diff

```ts
function Component() {
  const [name, setName] = useState<string>('React') // [!code --]
  const [name, setName] = useState('React') // [!code ++]
}
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim)

---

---
url: /catalog/tsx/rename-svg-attribute.md
---
## Rename SVG Attribute&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogcmV3cml0ZS1zdmctYXR0cmlidXRlXG5sYW5ndWFnZTogdHN4XG5ydWxlOlxuICBwYXR0ZXJuOiAkUFJPUFxuICByZWdleDogKFthLXpdKyktKFthLXpdKVxuICBraW5kOiBwcm9wZXJ0eV9pZGVudGlmaWVyXG4gIGluc2lkZTpcbiAgICBraW5kOiBqc3hfYXR0cmlidXRlXG50cmFuc2Zvcm06XG4gIE5FV19QUk9QOlxuICAgIGNvbnZlcnQ6XG4gICAgICBzb3VyY2U6ICRQUk9QXG4gICAgICB0b0Nhc2U6IGNhbWVsQ2FzZVxuZml4OiAkTkVXX1BST1AiLCJzb3VyY2UiOiJjb25zdCBlbGVtZW50ID0gKFxuICA8c3ZnIHdpZHRoPVwiMTAwXCIgaGVpZ2h0PVwiMTAwXCIgdmlld0JveD1cIjAgMCAxMDAgMTAwXCI+XG4gICAgPHBhdGggZD1cIk0xMCAyMCBMMzAgNDBcIiBzdHJva2UtbGluZWNhcD1cInJvdW5kXCIgZmlsbC1vcGFjaXR5PVwiMC41XCIgLz5cbiAgPC9zdmc+XG4pIn0=)

### Description

[SVG](https://en.wikipedia.org/wiki/SVG)(Scalable Vector Graphics)s' hyphenated names are not compatible with JSX syntax in React. JSX requires [camelCase naming](https://react.dev/learn/writing-markup-with-jsx#3-camelcase-salls-most-of-the-things) for attributes.
For example, an SVG attribute like `stroke-linecap` needs to be renamed to `strokeLinecap` to work correctly in React.

### YAML

```yaml
id: rewrite-svg-attribute
language: tsx
rule:
  pattern: $PROP            # capture in metavar
  regex: ([a-z]+)-([a-z])   # hyphenated name
  kind: property_identifier
  inside:
    kind: jsx_attribute     # in JSX attribute
transform:
  NEW_PROP:                 # new property name
    convert:                # use ast-grep's convert
      source: $PROP
      toCase: camelCase     # to camelCase naming
fix: $NEW_PROP
```

### Example

```tsx {3}
const element = (
  <svg width="100" height="100" viewBox="0 0 100 100">
    <path d="M10 20 L30 40" stroke-linecap="round" fill-opacity="0.5" />
  </svg>
)
```

### Diff

```ts
const element = (
  <svg width="100" height="100" viewBox="0 0 100 100">
    <path d="M10 20 L30 40" stroke-linecap="round" fill-opacity="0.5" /> // [!code --]
    <path d="M10 20 L30 40" strokeLinecap="round" fillOpacity="0.5" />   // [!code ++]
  </svg>
)
```

### Contributed by

Inspired by [SVG Renamer](https://admondtamang.medium.com/introducing-svg-renamer-your-solution-for-react-svg-attributes-26503382d5a8)

---

---
url: /catalog/tsx/rewrite-mobx-component.md
---
## Rewrite MobX Component Style&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoicnVsZTpcbiAgcGF0dGVybjogZXhwb3J0IGNvbnN0ICRDT01QID0gb2JzZXJ2ZXIoJEZVTkMpXG5maXg6IHwtXG4gIGNvbnN0IEJhc2UkQ09NUCA9ICRGVU5DXG4gIGV4cG9ydCBjb25zdCAkQ09NUCA9IG9ic2VydmVyKEJhc2UkQ09NUCkiLCJzb3VyY2UiOiJleHBvcnQgY29uc3QgRXhhbXBsZSA9IG9ic2VydmVyKCgpID0+IHtcbiAgcmV0dXJuIDxkaXY+SGVsbG8gV29ybGQ8L2Rpdj5cbn0pIn0=)

### Description

React and MobX are libraries that help us build user interfaces with JavaScript.

[React hooks](https://react.dev/reference/react) allow us to use state and lifecycle methods in functional components. But we need follow some hook rules, or React may break. [MobX](https://mobx.js.org/react-integration.html) has an `observer` function that makes a component update when data changes.

When we use the `observer` function like this:

```JavaScript
export const Example = observer(() => {…})
```

ESLint, the tool that checks hooks, thinks that `Example` is not a React component, but just a regular function. So it does not check the hooks inside it, and we may miss some wrong usages.

To fix this, we need to change our component style to this:

```JavaScript
const BaseExample = () => {…}
const Example = observer(BaseExample)
```

Now ESLint can see that `BaseExample` is a React component, and it can check the hooks inside it.

### YAML

```yaml
id: rewrite-mobx-component
language: typescript
rule:
  pattern: export const $COMP = observer($FUNC)
fix: |-
  const Base$COMP = $FUNC
  export const $COMP = observer(Base$COMP)
```

### Example

```js {1-3}
export const Example = observer(() => {
  return <div>Hello World</div>
})
```

### Diff

```js
export const Example = observer(() => { // [!code --]
  return <div>Hello World</div>         // [!code --]
})                                      // [!code --]
const BaseExample = () => {             // [!code ++]
  return <div>Hello World</div>         // [!code ++]
}                                       // [!code ++]
export const Example = observer(BaseExample) // [!code ++]
```

### Contributed by

[Bryan Lee](https://twitter.com/meetliby/status/1698601672568901723)

---

---
url: /catalog/tsx/unnecessary-react-hook.md
---
## Avoid Unnecessary React Hook

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6InV0aWxzOlxuICBob29rX2NhbGw6XG4gICAgaGFzOlxuICAgICAga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICByZWdleDogXnVzZVxuICAgICAgc3RvcEJ5OiBlbmRcbnJ1bGU6XG4gIGFueTpcbiAgLSBwYXR0ZXJuOiBmdW5jdGlvbiAkRlVOQygkJCQpIHsgJCQkIH1cbiAgLSBwYXR0ZXJuOiBsZXQgJEZVTkMgPSAoJCQkKSA9PiAkJCQgXG4gIC0gcGF0dGVybjogY29uc3QgJEZVTkMgPSAoJCQkKSA9PiAkJCRcbiAgaGFzOlxuICAgIHBhdHRlcm46ICRCT0RZXG4gICAga2luZDogc3RhdGVtZW50X2Jsb2NrXG4gICAgc3RvcEJ5OiBlbmQgXG5jb25zdHJhaW50czpcbiAgRlVOQzoge3JlZ2V4OiBedXNlIH1cbiAgQk9EWTogeyBub3Q6IHsgbWF0Y2hlczogaG9va19jYWxsIH0gfSBcbiIsInNvdXJjZSI6ImZ1bmN0aW9uIHVzZUlBbU5vdEhvb2tBY3R1YWxseShhcmdzKSB7XG4gICAgY29uc29sZS5sb2coJ0NhbGxlZCBpbiBSZWFjdCBidXQgSSBkb250IG5lZWQgdG8gYmUgYSBob29rJylcbiAgICByZXR1cm4gYXJncy5sZW5ndGhcbn1cbmNvbnN0IHVzZUlBbU5vdEhvb2tUb28gPSAoLi4uYXJncykgPT4ge1xuICAgIGNvbnNvbGUubG9nKCdDYWxsZWQgaW4gUmVhY3QgYnV0IEkgZG9udCBuZWVkIHRvIGJlIGEgaG9vaycpXG4gICAgcmV0dXJuIGFyZ3MubGVuZ3RoXG59XG5cbmZ1bmN0aW9uIHVzZUhvb2soKSB7XG4gICAgdXNlRWZmZWN0KCgpID0+IHtcbiAgICAgIGNvbnNvbGUubG9nKCdSZWFsIGhvb2snKSAgIFxuICAgIH0pXG59In0=)

### Description

React hook is a powerful feature in React that allows you to use state and other React features in a functional component.

However, you should avoid using hooks when you don't need them. If the code does not contain using any other React hooks,
it can be rewritten to a plain function. This can help to separate your application logic from the React-specific UI logic.

### YAML

```yaml
id: unnecessary-react-hook
language: Tsx
utils:
  hook_call:
    has:
      kind: call_expression
      regex: ^use
      stopBy: end
rule:
  any:
  - pattern: function $FUNC($$$) { $$$ }
  - pattern: let $FUNC = ($$$) => $$$
  - pattern: const $FUNC = ($$$) => $$$
  has:
    pattern: $BODY
    kind: statement_block
    stopBy: end
constraints:
  FUNC: {regex: ^use }
  BODY: { not: { matches: hook_call } }
```

### Example

```tsx {1-8}
function useIAmNotHookActually(args) {
    console.log('Called in React but I dont need to be a hook')
    return args.length
}
const useIAmNotHookToo = (...args) => {
    console.log('Called in React but I dont need to be a hook')
    return args.length
}

function useTrueHook() {
    useEffect(() => {
      console.log('Real hook')
    })
}
```

### Contributed by

[Herrington Darkholme](https://twitter.com/hd_nvim)

---

---
url: /catalog/tsx/reverse-react-compiler.md
---
## Reverse React Compiler™&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InRzeCIsInF1ZXJ5IjoiIiwicmV3cml0ZSI6IiIsInN0cmljdG5lc3MiOiJyZWxheGVkIiwic2VsZWN0b3IiOiIiLCJjb25maWciOiJpZDogcmV3cml0ZS1jYWNoZSBcbmxhbmd1YWdlOiB0c3hcbnJ1bGU6XG4gIGFueTpcbiAgLSBwYXR0ZXJuOiB1c2VDYWxsYmFjaygkRk4sICQkJClcbiAgLSBwYXR0ZXJuOiBtZW1vKCRGTiwgJCQkKVxuZml4OiAkRk5cblxuLS0tXG5cbmlkOiByZXdyaXRlLXVzZS1tZW1vXG5sYW5ndWFnZTogdHN4XG5ydWxlOiB7IHBhdHRlcm46ICd1c2VNZW1vKCRGTiwgJCQkKScgfVxuZml4OiAoJEZOKSgpIiwic291cmNlIjoiY29uc3QgQ29tcG9uZW50ID0gKCkgPT4ge1xuICBjb25zdCBbY291bnQsIHNldENvdW50XSA9IHVzZVN0YXRlKDApXG4gIGNvbnN0IGluY3JlbWVudCA9IHVzZUNhbGxiYWNrKCgpID0+IHtcbiAgICBzZXRDb3VudCgocHJldkNvdW50KSA9PiBwcmV2Q291bnQgKyAxKVxuICB9LCBbXSlcbiAgY29uc3QgZXhwZW5zaXZlQ2FsY3VsYXRpb24gPSB1c2VNZW1vKCgpID0+IHtcbiAgICAvLyBtb2NrIEV4cGVuc2l2ZSBjYWxjdWxhdGlvblxuICAgIHJldHVybiBjb3VudCAqIDJcbiAgfSwgW2NvdW50XSlcblxuICByZXR1cm4gKFxuICAgIDw+XG4gICAgICA8cD5FeHBlbnNpdmUgUmVzdWx0OiB7ZXhwZW5zaXZlQ2FsY3VsYXRpb259PC9wPlxuICAgICAgPGJ1dHRvbiBvbkNsaWNrPXtpbmNyZW1lbnR9Pntjb3VudH08L2J1dHRvbj5cbiAgICA8Lz5cbiAgKVxufSJ9)

### Description

React Compiler is a build-time only tool that automatically optimizes your React app, working with plain JavaScript and understanding the Rules of React without requiring a rewrite. It optimizes apps by automatically memoizing code, similar to `useMemo`, `useCallback`, and `React.memo`, reducing unnecessary recomputation due to incorrect or forgotten memoization.

Reverse React Compiler™ is a [parody tweet](https://x.com/aidenybai/status/1881397529369034997) that works in the opposite direction. It takes React code and removes memoization,  guaranteed to make your code slower. ([not](https://x.com/kentcdodds/status/1881404373646880997) [necessarily](https://dev.to/prathamisonline/are-you-over-using-usememo-and-usecallback-hooks-in-react-5lp))

It is originally written in Babel and this is an [ast-grep version](https://x.com/hd_nvim/status/1881402678493970620) of it.

:::details The Original Babel Implementation
For comparison purposes only. Note the original code [does not correctly rewrite](https://x.com/hd_nvim/status/1881404893136896415) `useMemo`.

```js
const ReverseReactCompiler = ({ types: t }) => ({
  visitor: {
    CallExpression(path) {
      const callee = path.node.callee;
      if (
        t.isIdentifier(callee, { name: "useMemo" }) ||
        t.isIdentifier(callee, { name: "useCallback" }) ||
        t.isIdentifier(callee, { name: "memo" })
      ) {
        path.replaceWith(args[0]);
      }
    },
  },
});
```

:::

### YAML

```yaml
id: rewrite-cache
language: tsx
rule:
  any:
  - pattern: useCallback($FN, $$$)
  - pattern: memo($FN, $$$)
fix: $FN
---
id: rewrite-use-memo
language: tsx
rule: { pattern: 'useMemo($FN, $$$)' }
fix: ($FN)()   # need IIFE to wrap memo function
```

### Example

```tsx {3-5,6-9}
const Component = () => {
  const [count, setCount] = useState(0)
  const increment = useCallback(() => {
    setCount((prevCount) => prevCount + 1)
  }, [])
  const expensiveCalculation = useMemo(() => {
    // mock Expensive calculation
    return count * 2
  }, [count])

  return (
    <>
      <p>Expensive Result: {expensiveCalculation}</p>
      <button onClick={increment}>{count}</button>
    </>
  )
}
```

### Diff

```tsx
const Component = () => {
  const [count, setCount] = useState(0)
  const increment = useCallback(() => {     // [!code --]
    setCount((prevCount) => prevCount + 1)  // [!code --]
  }, [])                                 // [!code --]
  const increment = () => {         // [!code ++]
    setCount((prevCount) => prevCount + 1) // [!code ++]
  } // [!code ++]
  const expensiveCalculation = useMemo(() => { // [!code --]
    // mock Expensive calculation             // [!code --]
    return count * 2                        // [!code --]
  }, [count])                             // [!code --]
  const expensiveCalculation = (() => { // [!code ++]
    // mock Expensive calculation      // [!code ++]
    return count * 2                 // [!code ++]
  })()                            // [!code ++]
  return (
    <>
      <p>Expensive Result: {expensiveCalculation}</p>
      <button onClick={increment}>{count}</button>
    </>
  )
}
```

### Contributed by

Inspired by [Aiden Bai](https://twitter.com/aidenybai)

---

---
url: /catalog/typescript/find-import-file-without-extension.md
---
## Find Import File without Extension

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoibGFuZ3VhZ2U6IFwianNcIlxucnVsZTpcbiAgcmVnZXg6IFwiL1teLl0rW14vXSRcIiAgXG4gIGtpbmQ6IHN0cmluZ19mcmFnbWVudFxuICBhbnk6XG4gICAgLSBpbnNpZGU6XG4gICAgICAgIHN0b3BCeTogZW5kXG4gICAgICAgIGtpbmQ6IGltcG9ydF9zdGF0ZW1lbnRcbiAgICAtIGluc2lkZTpcbiAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAgICAga2luZDogY2FsbF9leHByZXNzaW9uXG4gICAgICAgIGhhczpcbiAgICAgICAgICBmaWVsZDogZnVuY3Rpb25cbiAgICAgICAgICByZWdleDogXCJeaW1wb3J0JFwiXG4iLCJzb3VyY2UiOiJpbXBvcnQgYSwge2IsIGMsIGR9IGZyb20gXCIuL2ZpbGVcIjtcbmltcG9ydCBlIGZyb20gXCIuL290aGVyX2ZpbGUuanNcIjtcbmltcG9ydCBcIi4vZm9sZGVyL1wiO1xuaW1wb3J0IHt4fSBmcm9tIFwicGFja2FnZVwiO1xuaW1wb3J0IHt5fSBmcm9tIFwicGFja2FnZS93aXRoL3BhdGhcIjtcblxuaW1wb3J0KFwiLi9keW5hbWljMVwiKTtcbmltcG9ydChcIi4vZHluYW1pYzIuanNcIik7XG5cbm15X2Z1bmMoXCIuL3VucmVsYXRlZF9wYXRoX3N0cmluZ1wiKVxuXG4ifQ==)

### Description

In ECMAScript modules (ESM), the module specifier must include the file extension, such as `.js` or `.mjs`, when importing local or absolute modules. This is because ESM does not perform any automatic file extension resolution, unlike CommonJS modules tools such as Webpack or Babel. This behavior matches how import behaves in browser environments, and is specified by the [ESM module spec](https://stackoverflow.com/questions/66375075/node-14-ecmascript-modules-import-modules-without-file-extensions).

The rule finds all imports (static and dynamic) for files without a file extension.

### YAML

```yaml
id: find-import-file
language: js
rule:
  regex: "/[^.]+[^/]$"
  kind: string_fragment
  any:
    - inside:
        stopBy: end
        kind: import_statement
    - inside:
        stopBy: end
        kind: call_expression
        has:
          field: function
          regex: "^import$"
```

### Example

```ts {1,5,7}
import a, {b, c, d} from "./file";
import e from "./other_file.js";
import "./folder/";
import {x} from "package";
import {y} from "package/with/path";

import("./dynamic1");
import("./dynamic2.js");

my_func("./unrelated_path_string")
```

### Contributed by

[DasSurma](https://twitter.com/DasSurma) in [this tweet](https://x.com/DasSurma/status/1706213303331029277).

---

---
url: /catalog/typescript/find-import-usage.md
---
## Find Import Usage

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InR5cGVzY3JpcHQiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoicnVsZTpcbiAgIyB0aGUgdXNhZ2VcbiAga2luZDogaWRlbnRpZmllclxuICBwYXR0ZXJuOiAkTU9EXG4gICMgaXRzIHJlbGF0aW9uc2hpcCB0byB0aGUgcm9vdFxuICBpbnNpZGU6XG4gICAgc3RvcEJ5OiBlbmRcbiAgICBraW5kOiBwcm9ncmFtXG4gICAgIyBhbmQgYmFjayBkb3duIHRvIHRoZSBpbXBvcnQgc3RhdGVtZW50XG4gICAgaGFzOlxuICAgICAga2luZDogaW1wb3J0X3N0YXRlbWVudFxuICAgICAgIyBhbmQgZGVlcGVyIGludG8gdGhlIGltcG9ydCBzdGF0ZW1lbnQgbG9va2luZyBmb3IgdGhlIG1hdGNoaW5nIGlkZW50aWZpZXJcbiAgICAgIGhhczpcbiAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAgICAga2luZDogaW1wb3J0X3NwZWNpZmllclxuICAgICAgICBwYXR0ZXJuOiAkTU9EICMgc2FtZSBwYXR0ZXJuIGFzIHRoZSB1c2FnZSBpcyBlbmZvcmNlZCBoZXJlIiwic291cmNlIjoiaW1wb3J0IHsgTW9uZ29DbGllbnQgfSBmcm9tICdtb25nb2RiJztcbmNvbnN0IHVybCA9ICdtb25nb2RiOi8vbG9jYWxob3N0OjI3MDE3JztcbmFzeW5jIGZ1bmN0aW9uIHJ1bigpIHtcbiAgY29uc3QgY2xpZW50ID0gbmV3IE1vbmdvQ2xpZW50KHVybCk7XG59XG4ifQ==)

### Description

It is common to find the usage of an imported module in a codebase. This rule helps you to find the usage of an imported module in your codebase.
The idea of this rule can be broken into several parts:

* Find the use of an identifier `$MOD`
* To find the import, we first need to find the root file of which `$MOD` is  `inside`
* The `program` file `has` an `import` statement
* The `import` statement `has` the identifier `$MOD`

### YAML

```yaml
id: find-import-usage
language: typescript
rule:
  kind: identifier # ast-grep requires a kind
  pattern: $MOD   # the identifier to find
  inside: # find the root
    stopBy: end
    kind: program
    has: # and has the import statement
      kind: import_statement
      has: # look for the matching identifier
        stopBy: end
        kind: import_specifier
        pattern: $MOD # same pattern as the usage is enforced here
```

### Example

```ts {4}
import { MongoClient } from 'mongodb';
const url = 'mongodb://localhost:27017';
async function run() {
  const client = new MongoClient(url);
}
```

### Contributed by

[Steven Love](https://github.com/StevenLove)

---

---
url: /catalog/typescript/missing-component-decorator.md
---
## Missing Component Decorator

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImltcG9ydCAkQSBmcm9tICdhbmltZWpzJyIsInJld3JpdGUiOiJpbXBvcnQgeyBhbmltZSBhcyAkQSB9IGZyb20gJ2FuaW1lJyIsInN0cmljdG5lc3MiOiJzbWFydCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IG1pc3NpbmctY29tcG9uZW50LWRlY29yYXRvclxubWVzc2FnZTogWW91J3JlIHVzaW5nIGFuIEFuZ3VsYXIgbGlmZWN5Y2xlIG1ldGhvZCwgYnV0IG1pc3NpbmcgYW4gQW5ndWxhciBAQ29tcG9uZW50KCkgZGVjb3JhdG9yLlxubGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbnNldmVyaXR5OiB3YXJuaW5nXG5ydWxlOlxuICBwYXR0ZXJuOlxuICAgIGNvbnRleHQ6ICdjbGFzcyBIaSB7ICRNRVRIT0QoKSB7ICQkJF99IH0nXG4gICAgc2VsZWN0b3I6IG1ldGhvZF9kZWZpbml0aW9uXG4gIGluc2lkZTpcbiAgICBwYXR0ZXJuOiAnY2xhc3MgJEtMQVNTICQkJF8geyAkJCRfIH0nXG4gICAgc3RvcEJ5OiBlbmRcbiAgICBub3Q6XG4gICAgICBoYXM6XG4gICAgICAgIHBhdHRlcm46ICdAQ29tcG9uZW50KCQkJF8pJ1xuY29uc3RyYWludHM6XG4gIE1FVEhPRDpcbiAgICByZWdleDogbmdPbkluaXR8bmdPbkRlc3Ryb3lcbmxhYmVsczpcbiAgS0xBU1M6XG4gICAgc3R5bGU6IHByaW1hcnlcbiAgICBtZXNzYWdlOiBcIlRoaXMgY2xhc3MgaXMgbWlzc2luZyB0aGUgZGVjb3JhdG9yLlwiXG4gIE1FVEhPRDpcbiAgICBzdHlsZTogc2Vjb25kYXJ5XG4gICAgbWVzc2FnZTogXCJUaGlzIGlzIGFuIEFuZ3VsYXIgbGlmZWN5Y2xlIG1ldGhvZC5cIlxubWV0YWRhdGE6XG4gIGNvbnRyaWJ1dGVkQnk6IHNhbXdpZ2h0dCIsInNvdXJjZSI6ImNsYXNzIE5vdENvbXBvbmVudCB7XG4gICAgbmdPbkluaXQoKSB7fVxufVxuXG5AQ29tcG9uZW50KClcbmNsYXNzIEtsYXNzIHtcbiAgICBuZ09uSW5pdCgpIHt9XG59In0=)

### Description

Angular lifecycle methods are a set of methods that allow you to hook into the lifecycle of an Angular component or directive.
They must be used within a class that is decorated with the `@Component()` decorator.

### YAML

This rule illustrates how to use custom labels to highlight specific parts of the code.

```yaml
id: missing-component-decorator
message: You're using an Angular lifecycle method, but missing an Angular @Component() decorator.
language: TypeScript
severity: warning
rule:
  pattern:
    context: 'class Hi { $METHOD() { $$$_} }'
    selector: method_definition
  inside:
    pattern: 'class $KLASS $$$_ { $$$_ }'
    stopBy: end
    not:
      has:
        pattern: '@Component($$$_)'
constraints:
  METHOD:
    regex: ngOnInit|ngOnDestroy
labels:
  KLASS:
    style: primary
    message: "This class is missing the decorator."
  METHOD:
    style: secondary
    message: "This is an Angular lifecycle method."
metadata:
  contributedBy: samwightt
```

### Example

```ts {2}
class NotComponent {
    ngOnInit() {}
}

@Component()
class Klass {
    ngOnInit() {}
}
```

### Contributed by

[Sam Wight](https://github.com/samwightt).

---

---
url: /catalog/typescript/no-await-in-promise-all.md
---
## No `await` in `Promise.all` array&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImNvbnNvbGUubG9nKCRNQVRDSCkiLCJyZXdyaXRlIjoibG9nZ2VyLmxvZygkTUFUQ0gpIiwiY29uZmlnIjoiaWQ6IG5vLWF3YWl0LWluLXByb21pc2UtYWxsXG5zZXZlcml0eTogZXJyb3Jcbmxhbmd1YWdlOiBKYXZhU2NyaXB0XG5tZXNzYWdlOiBObyBhd2FpdCBpbiBQcm9taXNlLmFsbFxucnVsZTpcbiAgcGF0dGVybjogYXdhaXQgJEFcbiAgaW5zaWRlOlxuICAgIHBhdHRlcm46IFByb21pc2UuYWxsKCRfKVxuICAgIHN0b3BCeTpcbiAgICAgIG5vdDogeyBhbnk6IFt7a2luZDogYXJyYXl9LCB7a2luZDogYXJndW1lbnRzfV0gfVxuZml4OiAkQSIsInNvdXJjZSI6ImNvbnN0IFtmb28sIGJhcl0gPSBhd2FpdCBQcm9taXNlLmFsbChbXG4gIGF3YWl0IGdldEZvbygpLFxuICBnZXRCYXIoKSxcbiAgKGFzeW5jICgpID0+IHsgYXdhaXQgZ2V0QmF6KCl9KSgpLFxuXSkifQ==)

### Description

Using `await` inside an inline `Promise.all` array is usually a mistake, as it defeats the purpose of running the promises in parallel. Instead, the promises should be created without `await` and passed to `Promise.all`, which can then be awaited.

### YAML

```yaml
id: no-await-in-promise-all
language: typescript
rule:
  pattern: await $A
  inside:
    pattern: Promise.all($_)
    stopBy:
      not: { any: [{kind: array}, {kind: arguments}] }
fix: $A
```

### Example

```ts {2}
const [foo, bar] = await Promise.all([
  await getFoo(),
  getBar(),
  (async () => { await getBaz()})(),
])
```

### Diff

```ts
const [foo, bar] = await Promise.all([
  await getFoo(), // [!code --]
  getFoo(), // [!code ++]
  getBar(),
  (async () => { await getBaz()})(),
])
```

### Contributed by

Inspired by [Alvar Lagerlöf](https://twitter.com/alvarlagerlof)

---

---
url: /catalog/typescript/migrate-xstate-v5.md
---
## Migrate XState to v5 from v4&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImlmICgkQSkgeyAkJCRCIH0iLCJyZXdyaXRlIjoiaWYgKCEoJEEpKSB7XG4gICAgcmV0dXJuO1xufVxuJCQkQiIsImNvbmZpZyI6InV0aWxzOlxuICBGUk9NX1hTVEFURTogeyBraW5kOiBpbXBvcnRfc3RhdGVtZW50LCBoYXM6IHsga2luZDogc3RyaW5nLCByZWdleDogeHN0YXRlIH0gfVxuICBYU1RBVEVfRVhQT1JUOlxuICAgIGtpbmQ6IGlkZW50aWZpZXJcbiAgICBpbnNpZGU6IHsgaGFzOiB7IG1hdGNoZXM6IEZST01fWFNUQVRFIH0sIHN0b3BCeTogZW5kIH1cbnJ1bGU6IHsgcmVnZXg6IF5NYWNoaW5lfGludGVycHJldCQsIHBhdHRlcm46ICRJTVBPUlQsIG1hdGNoZXM6IFhTVEFURV9FWFBPUlQgfVxudHJhbnNmb3JtOlxuICBTVEVQMTogXG4gICAgcmVwbGFjZToge2J5OiBjcmVhdGUkMSwgcmVwbGFjZTogKE1hY2hpbmUpLCBzb3VyY2U6ICRJTVBPUlQgfVxuICBGSU5BTDpcbiAgICByZXBsYWNlOiB7IGJ5OiBjcmVhdGVBY3RvciwgcmVwbGFjZTogaW50ZXJwcmV0LCBzb3VyY2U6ICRTVEVQMSB9XG5maXg6ICRGSU5BTFxuLS0tIFxucnVsZTogeyBwYXR0ZXJuOiAkTUFDSElORS53aXRoQ29uZmlnIH1cbmZpeDogJE1BQ0hJTkUucHJvdmlkZVxuLS0tXG5ydWxlOlxuICBraW5kOiBwcm9wZXJ0eV9pZGVudGlmaWVyXG4gIHJlZ2V4OiBec2VydmljZXMkXG4gIGluc2lkZTogeyBwYXR0ZXJuOiAgJE0ud2l0aENvbmZpZygkJCRBUkdTKSwgc3RvcEJ5OiBlbmQgfVxuZml4OiBhY3RvcnMiLCJzb3VyY2UiOiJpbXBvcnQgeyBNYWNoaW5lLCBpbnRlcnByZXQgfSBmcm9tICd4c3RhdGUnO1xuXG5jb25zdCBtYWNoaW5lID0gTWFjaGluZSh7IC8qLi4uKi99KTtcblxuY29uc3Qgc3BlY2lmaWNNYWNoaW5lID0gbWFjaGluZS53aXRoQ29uZmlnKHtcbiAgYWN0aW9uczogeyAvKiAuLi4gKi8gfSxcbiAgZ3VhcmRzOiB7IC8qIC4uLiAqLyB9LFxuICBzZXJ2aWNlczogeyAvKiAuLi4gKi8gfSxcbn0pO1xuXG5jb25zdCBhY3RvciA9IGludGVycHJldChzcGVjaWZpY01hY2hpbmUsIHtcbi8qIGFjdG9yIG9wdGlvbnMgKi9cbn0pOyJ9)

### Description

[XState](https://xstate.js.org/) is a state management/orchestration library based on state machines, statecharts, and the actor model. It allows you to model complex logic in event-driven ways, and orchestrate the behavior of many actors communicating with each other.

XState's v5 version introduced some breaking changes and new features compared to v4.
While the migration should be a straightforward process, it is a tedious process and requires knowledge of the differences between v4 and v5.

ast-grep provides a way to automate the process and a way to encode valuable knowledge to executable rules.

The following example picks up some migration items and demonstrates the power of ast-grep's rule system.

### YAML

The rules below correspond to XState v5's [`createMachine`](https://stately.ai/docs/migration#use-createmachine-not-machine), [`createActor`](https://stately.ai/docs/migration#use-createactor-not-interpret), and [`machine.provide`](https://stately.ai/docs/migration#use-machineprovide-not-machinewithconfig).

The example shows how ast-grep can use various features like [utility rule](/guide/rule-config/utility-rule.html), [transformation](/reference/yaml/transformation.html) and [multiple rule in single file](/reference/playground.html#test-multiple-rules) to automate the migration. Each rule has a clear and descriptive `id` field that explains its purpose.

For more information, you can use [Codemod AI](https://app.codemod.com/studio?ai_thread_id=new) to provide more detailed explanation for each rule.

```yaml
id: migrate-import-name
utils:
  FROM_XS: {kind: import_statement, has: {kind: string, regex: xstate}}
  XS_EXPORT:
    kind: identifier
    inside: { has: { matches: FROM_XS }, stopBy: end }
rule: { regex: ^Machine|interpret$, pattern: $IMPT, matches: XS_EXPORT }
transform:
  STEP1:
    replace: {by: create$1, replace: (Machine), source: $IMPT }
  FINAL:
    replace: { by: createActor, replace: interpret, source: $STEP1 }
fix: $FINAL

---

id: migrate-to-provide
rule: { pattern: $MACHINE.withConfig }
fix: $MACHINE.provide

---

id: migrate-to-actors
rule:
  kind: property_identifier
  regex: ^services$
  inside: { pattern:  $M.withConfig($$$ARGS), stopBy: end }
fix: actors
```

### Example

```js {1,3,5,8,11}
import { Machine, interpret } from 'xstate';

const machine = Machine({ /*...*/});

const specificMachine = machine.withConfig({
  actions: { /* ... */ },
  guards: { /* ... */ },
  services: { /* ... */ },
});

const actor = interpret(specificMachine, {
  /* actor options */
});
```

### Diff

```js
import { Machine, interpret } from 'xstate'; // [!code --]
import { createMachine, createActor } from 'xstate'; // [!code ++]

const machine = Machine({ /*...*/}); // [!code --]
const machine = createMachine({ /*...*/}); // [!code ++]

const specificMachine = machine.withConfig({ // [!code --]
const specificMachine = machine.provide({ // [!code ++]
  actions: { /* ... */ },
  guards: { /* ... */ },
  services: { /* ... */ }, // [!code --]
  actors: { /* ... */ }, // [!code ++]
});

const actor = interpret(specificMachine, { // [!code --]
const actor = createActor(specificMachine, { // [!code ++]
  /* actor options */
});
```

### Contributed by

Inspired by [XState's blog](https://stately.ai/blog/2023-12-01-xstate-v5).

---

---
url: /catalog/typescript/no-console-except-catch.md
---
## No `console` except in `catch` block&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6ImphdmFzY3JpcHQiLCJxdWVyeSI6ImlmICRBLmhhc19mZWF0dXJlP1xuICAgICQkJEJcbmVsc2UgXG4gICAgJCQkQyBcbmVuZCAiLCJyZXdyaXRlIjoiJCQkQiIsImNvbmZpZyI6InJ1bGU6XG4gIGFueTpcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUuZXJyb3IoJCQkKVxuICAgICAgbm90OlxuICAgICAgICBpbnNpZGU6XG4gICAgICAgICAga2luZDogY2F0Y2hfY2xhdXNlXG4gICAgICAgICAgc3RvcEJ5OiBlbmRcbiAgICAtIHBhdHRlcm46IGNvbnNvbGUuJE1FVEhPRCgkJCQpXG5jb25zdHJhaW50czpcbiAgTUVUSE9EOlxuICAgIHJlZ2V4OiAnbG9nfGRlYnVnfHdhcm4nXG5maXg6ICcnIiwic291cmNlIjoiY29uc29sZS5kZWJ1ZygnJylcbnRyeSB7XG4gICAgY29uc29sZS5sb2coJ2hlbGxvJylcbn0gY2F0Y2ggKGUpIHtcbiAgICBjb25zb2xlLmVycm9yKGUpXG59In0=)

### Description

Using `console` methods is usually for debugging purposes and therefore not suitable to ship to the client.
`console` can expose sensitive information, clutter the output, or affect the performance.

The only exception is using `console.error` to log errors in the catch block, which can be useful for debugging production.

### YAML

```yaml
id: no-console-except-error
language: typescript
rule:
  any:
    - pattern: console.error($$$)
      not:
        inside:
          kind: catch_clause
          stopBy: end
    - pattern: console.$METHOD($$$)
constraints:
  METHOD:
    regex: 'log|debug|warn'
```

### Example

```ts {1,3}
console.debug('')
try {
    console.log('hello')
} catch (e) {
    console.error(e) // OK
}
```

### Diff

```ts
console.debug('') // [!code --]
try {
    console.log('hello') // [!code --]
} catch (e) {
    console.error(e) // OK
}
```

### Contributed by

Inspired by [Jerry Mouse](https://github.com/WWK563388548)

---

---
url: /catalog/typescript/switch-from-should-to-expect.md
---
## Switch Chai from `should` style to `expect`&#x20;

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InJ1c3QiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoicmVsYXhlZCIsInNlbGVjdG9yIjoiIiwiY29uZmlnIjoiaWQ6IHNob3VsZF90b19leHBlY3RfaW5zdGFuY2VvZlxubGFuZ3VhZ2U6IFR5cGVTY3JpcHRcbnJ1bGU6XG4gIGFueTpcbiAgLSBwYXR0ZXJuOiAkTkFNRS5zaG91bGQuYmUuYW4uaW5zdGFuY2VvZigkVFlQRSlcbiAgLSBwYXR0ZXJuOiAkTkFNRS5zaG91bGQuYmUuYW4uaW5zdGFuY2VPZigkVFlQRSlcbmZpeDogfC1cbiAgZXhwZWN0KCROQU1FKS5pbnN0YW5jZU9mKCRUWVBFKVxuLS0tXG5pZDogc2hvdWxkX3RvX2V4cGVjdF9nZW5lcmljU2hvdWxkQmVcbmxhbmd1YWdlOiBUeXBlU2NyaXB0XG5ydWxlOlxuICBwYXR0ZXJuOiAkTkFNRS5zaG91bGQuYmUuJFBST1BcbmZpeDogfC1cbiAgZXhwZWN0KCROQU1FKS50by5iZS4kUFJPUFxuIiwic291cmNlIjoiaXQoJ3Nob3VsZCBwcm9kdWNlIGFuIGluc3RhbmNlIG9mIGNob2tpZGFyLkZTV2F0Y2hlcicsICgpID0+IHtcbiAgd2F0Y2hlci5zaG91bGQuYmUuYW4uaW5zdGFuY2VvZihjaG9raWRhci5GU1dhdGNoZXIpO1xufSk7XG5pdCgnc2hvdWxkIGV4cG9zZSBwdWJsaWMgQVBJIG1ldGhvZHMnLCAoKSA9PiB7XG4gIHdhdGNoZXIub24uc2hvdWxkLmJlLmEoJ2Z1bmN0aW9uJyk7XG4gIHdhdGNoZXIuZW1pdC5zaG91bGQuYmUuYSgnZnVuY3Rpb24nKTtcbiAgd2F0Y2hlci5hZGQuc2hvdWxkLmJlLmEoJ2Z1bmN0aW9uJyk7XG4gIHdhdGNoZXIuY2xvc2Uuc2hvdWxkLmJlLmEoJ2Z1bmN0aW9uJyk7XG4gIHdhdGNoZXIuZ2V0V2F0Y2hlZC5zaG91bGQuYmUuYSgnZnVuY3Rpb24nKTtcbn0pOyJ9)

### Description

[Chai](https://www.chaijs.com) is a BDD / TDD assertion library for JavaScript. It comes with [two styles](https://www.chaijs.com/) of assertions: `should` and `expect`.

The `expect` interface provides a function as a starting point for chaining your language assertions and works with `undefined` and `null` values.
The `should` style allows for the same chainable assertions as the expect interface, however it extends each object with a should property to start your chain and [does not work](https://www.chaijs.com/guide/styles/#should-extras) with `undefined` and `null` values.

This rule migrates Chai `should` style assertions to `expect` style assertions. Note this is an example rule and a excerpt from [the original rules](https://github.com/43081j/codemods/blob/cddfe101e7f759e4da08b7e2f7bfe892c20f6f48/codemods/chai-should-to-expect.yml).

### YAML

```yaml
id: should_to_expect_instanceof
language: TypeScript
rule:
  any:
  - pattern: $NAME.should.be.an.instanceof($TYPE)
  - pattern: $NAME.should.be.an.instanceOf($TYPE)
fix: |-
  expect($NAME).instanceOf($TYPE)
---
id: should_to_expect_genericShouldBe
language: TypeScript
rule:
  pattern: $NAME.should.be.$PROP
fix: |-
  expect($NAME).to.be.$PROP
```

### Example

```js {2,5-9}
it('should produce an instance of chokidar.FSWatcher', () => {
  watcher.should.be.an.instanceof(chokidar.FSWatcher);
});
it('should expose public API methods', () => {
  watcher.on.should.be.a('function');
  watcher.emit.should.be.a('function');
  watcher.add.should.be.a('function');
  watcher.close.should.be.a('function');
  watcher.getWatched.should.be.a('function');
});
```

### Diff

```js
it('should produce an instance of chokidar.FSWatcher', () => {
  watcher.should.be.an.instanceof(chokidar.FSWatcher); // [!code --]
  expect(watcher).instanceOf(chokidar.FSWatcher); // [!code ++]
});
it('should expose public API methods', () => {
  watcher.on.should.be.a('function');   // [!code --]
  watcher.emit.should.be.a('function'); // [!code --]
  watcher.add.should.be.a('function');  // [!code --]
  watcher.close.should.be.a('function'); // [!code --]
  watcher.getWatched.should.be.a('function'); // [!code --]
  expect(watcher.on).to.be.a('function'); // [!code ++]
  expect(watcher.emit).to.be.a('function'); // [!code ++]
  expect(watcher.add).to.be.a('function'); // [!code ++]
  expect(watcher.close).to.be.a('function'); // [!code ++]
  expect(watcher.getWatched).to.be.a('function'); // [!code ++]
});
```

### Contributed by

[James](https://bsky.app/profile/43081j.com), by [this post](https://bsky.app/profile/43081j.com/post/3lgimzfxza22i)

### Exercise

Exercise left to the reader: can you write a rule to implement [this migration to `node:assert`](https://github.com/paulmillr/chokidar/pull/1409/files)?

---

---
url: /catalog/typescript/use-logical-assignment.md
---
## Use Logical Assignment Operators

* [Playground Link](/playground.html#eyJtb2RlIjoiUGF0Y2giLCJsYW5nIjoiamF2YXNjcmlwdCIsInF1ZXJ5IjoiJEEgPSAkQSB8fCAkQiIsInJld3JpdGUiOiIkQSB8fD0gJEIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6IiMgWUFNTCBSdWxlIGlzIG1vcmUgcG93ZXJmdWwhXG4jIGh0dHBzOi8vYXN0LWdyZXAuZ2l0aHViLmlvL2d1aWRlL3J1bGUtY29uZmlnLmh0bWwjcnVsZVxucnVsZTpcbiAgYW55OlxuICAgIC0gcGF0dGVybjogY29uc29sZS5sb2coJEEpXG4gICAgLSBwYXR0ZXJuOiBjb25zb2xlLmRlYnVnKCRBKVxuZml4OlxuICBsb2dnZXIubG9nKCRBKSIsInNvdXJjZSI6ImNvbnN0IGEgPSB7IGR1cmF0aW9uOiA1MCwgdGl0bGU6ICcnIH07XG5cbmEuZHVyYXRpb24gPSBhLmR1cmF0aW9uIHx8IDEwO1xuY29uc29sZS5sb2coYS5kdXJhdGlvbik7XG5hLnRpdGxlID0gYS50aXRsZSB8fCAndGl0bGUgaXMgZW1wdHkuJztcbmNvbnNvbGUubG9nKGEudGl0bGUpO1xuIn0=)

### Description

A logical assignment operator in JavaScript combines a logical operation (like OR or nullish coalescing) with an assignment. It updates a variable or property only under specific conditions, making code more concise.

This is a relatively new feature in JavaScript (introduced in ES2021), so older codebases might not use it yet. This rule identifies instances where a variable is assigned a value using a logical OR (`||`) operation and suggests replacing it with the more concise logical assignment operator (`||=`).

### Pattern

```shell
ast-grep -p '$A = $A || $B' -r '$A ||= $B' -l ts
```

### Example

```ts {3,5}
const a = { duration: 50, title: '' };

a.duration = a.duration || 10;
console.log(a.duration);
a.title = a.title || 'title is empty.';
console.log(a.title);
```

### Diff

```ts
const a = { duration: 50, title: '' };

a.duration = a.duration || 10; // [!code --]
a.duration ||= 10; // [!code ++]
console.log(a.duration);
a.title = a.title || 'title is empty.'; // [!code --]
a.title ||= 'title is empty.'; // [!code ++]
console.log(a.title);
```

### Contributed by

Inspired by [this tweet](https://x.com/YTCodeAntonio/status/1973720331656605809).

---

---
url: /catalog/yaml/find-key-value.md
---
## Find key/value and Show Message

* [Playground Link](/playground.html#eyJtb2RlIjoiQ29uZmlnIiwibGFuZyI6InlhbWwiLCJxdWVyeSI6IiIsInJld3JpdGUiOiIiLCJzdHJpY3RuZXNzIjoic21hcnQiLCJzZWxlY3RvciI6IiIsImNvbmZpZyI6ImlkOiBkZXRlY3QtaG9zdC1wb3J0XG5tZXNzYWdlOiBZb3UgYXJlIHVzaW5nICRIT1NUIG9uIFBvcnQgJFBPUlQsIHBsZWFzZSBjaGFuZ2UgaXQgdG8gODAwMFxuc2V2ZXJpdHk6IGVycm9yXG5ydWxlOlxuICBhbnk6XG4gIC0gcGF0dGVybjogfFxuICAgICBwb3J0OiAkUE9SVFxuICAtIHBhdHRlcm46IHxcbiAgICAgaG9zdDogJEhPU1QiLCJzb3VyY2UiOiJkYjpcbiAgIHVzZXJuYW1lOiByb290XG4gICBwYXNzd29yZDogcm9vdFxuXG5zZXJ2ZXI6XG4gIGhvc3Q6IDEyNy4wLjAuMVxuICBwb3J0OiA4MDAxIn0=)

### Description

This YAML rule helps detecting specific host and port configurations in your code. For example, it checks if the port is set to something other than 8000 or if a particular host is used. It provides an error message prompting you to update the configuration.

### YAML

```yaml
id: detect-host-port
message: You are using $HOST on Port $PORT, please change it to 8000
severity: error
rule:
  any:
    - pattern: |
        port: $PORT
    - pattern: |
        host: $HOST
```

### Example

```yaml {5,6}
db:
  username: root
  password: root
server:
  host: 127.0.0.1
  port: 8001
```

### Contributed by

[rohitcoder](https://twitter.com/rohitcoder) on [Discord](https://discord.com/invite/4YZjf6htSQ).
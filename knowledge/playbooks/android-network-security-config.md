---
type: Playbook
title: "Configuring Android Network Security Config for Certificate Pinning"
description: "Declarative implementation of public key pinning in Android using network_security_config.xml, pin-set definitions, and backup/expiration policies."
tags: [android, security, network-security-config, xml, pin-set, playbook]
generated:
  by: "antigravity/1.0"
  at: "2026-09-14T09:58:11+05:30"
status: stable
sources:
  - id: video-guide
    resource: "Technical Guide: Certificate Pinning in Android Applications"
---

# Configuring Android Network Security Config for Certificate Pinning

This playbook details how to configure declarative certificate pinning using Android's Network Security Configuration framework introduced in Android 7.0 (API level 24).[^video-guide]

## Prerequisites

1. Target Android API level 24 or higher (`minSdk 24` or target SDK 24+).
2. Base64-encoded SHA-256 digests of the primary server's Subject Public Key Info (SPKI).
3. At least one backup key pair stored securely offline (cold storage) and its corresponding SPKI digest.

## Step 1: Create the Network Security Configuration XML

Create an XML file at `res/xml/network_security_config.xml` in your Android project resources directory.[^video-guide]

```xml
<?xml version="1.0" encoding="utf-8"?>
<network-security-config>
    <domain-config>
        <!-- Specify the targeted domain, optionally including subdomains -->
        <domain includeSubdomains="true">api.example.com</domain>
        
        <!-- Define pin set with safety expiration date -->
        <pin-set expiration="2027-01-01">
            <!-- Primary production key pin (Base64-encoded SHA-256 SPKI digest) -->
            <pin digest="SHA-256">7HIpactkIAq2Y49orFOOQKurWxmmSFZhBCoQYcRhJ3Y=</pin>
            
            <!-- Mandatory backup key pin to prevent client lockout -->
            <pin digest="SHA-256">2bUq7X2lA3xV4jV5w8kQ4rZ2y5j6l1x2c3v4b5n6m7=</pin>
        </pin-set>
    </domain-config>
</network-security-config>
```

### Key Elements

* `<domain-config>`: Scopes the security rules to specific hostnames. Multiple domains can be grouped under one configuration.
* `<pin-set expiration="YYYY-MM-DD">`: Defines the cryptographic pins and an mandatory expiration date. After this date, Android disables pinning enforcement for this set and falls back to standard CA trust, preventing permanent client bricking if rotation fails.[^video-guide]
* `<pin digest="SHA-256">`: Contains the Base64 SHA-256 digest of the Subject Public Key Info (SPKI).

## Step 2: Bind Configuration in AndroidManifest.xml

Link the resource file in your `AndroidManifest.xml` under the `<application>` tag using the `android:networkSecurityConfig` attribute.[^video-guide]

```xml
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.secureapp">

    <uses-permission android:name="android.permission.INTERNET" />

    <application
        android:allowBackup="false"
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name"
        android:networkSecurityConfig="@xml/network_security_config"
        android:theme="@style/AppTheme">
        
        <!-- Application components -->
    </application>
</manifest>
```

## Step 3: Generating SPKI Hashes from Server Certificates

To obtain the Base64-encoded SHA-256 hash directly from an active endpoint or PEM certificate:

```bash
# Option A: Extract from live server
openssl s_client -servername api.example.com -connect api.example.com:443 </dev/null \
  | openssl x509 -pubkey -noout \
  | openssl pkey -pubin -outform der \
  | openssl dgst -sha256 -binary \
  | openssl enc -base64

# Option B: Extract from a local certificate file
openssl x509 -in server.crt -pubkey -noout \
  | openssl pkey -pubin -outform der \
  | openssl dgst -sha256 -binary \
  | openssl enc -base64
```

## Step 4: Lockout Prevention and Safety Invariants

To avoid stranding users when credentials rotate, enforce these invariants:[^video-guide]

1. **Always provide at least 2 pins**: One for the current active production key and at least one for an offline backup key.
2. **Set a rolling expiration date**: Set `expiration` to approximately 60–90 days past the planned key rotation window. Update the expiration date with standard application releases.
3. **Debug overrides for testing**: Do not disable pinning in production builds. For local proxy inspection during QA, use debug-specific trust anchors in `network_security_config.xml`:

```xml
<debug-overrides>
    <trust-anchors>
        <!-- Trust user-installed certs (e.g., Charles Proxy / mitmproxy) only in debug builds -->
        <certificates src="user" />
    </trust-anchors>
</debug-overrides>
```

## Related Topics

* [Android Certificate Pinning Mechanics](../concepts/android-certificate-pinning.md): Underlying threat model and cryptographic design.
* [Video Guide Reference](../references/android-certificate-pinning-guide.md): Source summary and video breakdown.

## Deep Internals

1. **OkHttp vs. Network Security Config Precedence**: If an Android application uses OkHttp's programmatic `CertificatePinner` simultaneously with Android's XML `network_security_config.xml`, both layers run independently. If either layer rejects the server certificate chain, the connection terminates. However, OkHttp's `CertificatePinner` does not automatically inherit the XML `<pin-set expiration="...">` policy unless custom logic is supplied.
2. **Clean Certificate Chain Reconstruction**: Android's pinning mechanism evaluates certificates only after the trust manager successfully builds and validates an authentic chain to a trusted CA root. If a self-signed or invalid certificate is presented, TLS handshake failure occurs during standard path building before pin matching is even invoked.
3. **HPKP / RFC 7469 Heritage**: Android's `<pin digest="SHA-256">` format is directly derived from HTTP Public Key Pinning (HPKP RFC 7469). While HPKP was deprecated in web browsers due to lockout risks, Android preserved the declarative format within native app configuration where developers control the client release cycle.

[^video-guide]: Technical Guide: Certificate Pinning in Android Applications (Chapters 1-3).

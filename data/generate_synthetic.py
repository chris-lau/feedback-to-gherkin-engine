#!/usr/bin/env python3
"""Synthetic feedback -> Gherkin dataset generator.

Two sources:

  local (default)  Deterministic, seeded composition from hand-authored domain
                   templates. Runs offline, no API key, fully reproducible.
  api              Teacher-model distillation (DeepSeek / any OpenAI-compatible
                   endpoint) prompted with the same domain archetypes. Requires
                   TEACHER_API_KEY (and optionally TEACHER_BASE_URL,
                   TEACHER_MODEL) in the environment.

Each emitted record: {"id", "domain", "input", "output"} where `input` is messy
first-person user feedback and `output` is the strict canonical schema:

    PROBLEM STATEMENT: ...
    USER STORY: As a ..., I want ..., So that ...
    ACCEPTANCE CRITERIA:
    Scenario: ...
    Given ...
    When ...
    Then ...
    And ...

Downstream, data/lint_dataset.py validates every record before the 80/20 split.

Usage:
    python data/generate_synthetic.py --source local --n 250 --seed 42 \
        --out data/raw/generated.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical system prompt (kept in sync with notebooks/, eval/, backend/)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a requirements formatting engine. Convert the raw user feedback "
    "into exactly three sections in this order: 'PROBLEM STATEMENT:', "
    "'USER STORY:', 'ACCEPTANCE CRITERIA:'. The user story must follow the "
    "pattern 'As a ..., I want ..., So that ...'. Acceptance criteria must use "
    "strict Gherkin (Scenario:, Given, When, Then, And). Output only the three "
    "sections. No greetings, no explanations, no markdown."
)

# ---------------------------------------------------------------------------
# Domain definitions: 5 domains x 8 friction archetypes = 40 archetypes.
# `frictions` fields:
#   symptoms  - paraphrase pool for the messy input description
#   details   - optional concrete garnish (error strings, versions, counts)
#   impacts   - business-impact paraphrase pool for the messy input
#   problem   - canonical PROBLEM STATEMENT template ({role} is substituted)
#   want / so_that / scenario / given / when / then / ands - canonical schema
# ---------------------------------------------------------------------------

DOMAINS = [
    {
        "code": "auth",
        "name": "Enterprise Auth / SSO",
        "roles": [
            "IT operations lead at a mid-size fintech",
            "helpdesk technician at a regional healthcare group",
            "DevOps engineer on an internal tools team",
            "security administrator at a logistics company",
            "workspace tenancy administrator at a B2B SaaS company",
            "engineering manager at a startup scaling past 200 employees",
        ],
        "systems": [
            "the SSO portal",
            "the Okta integration",
            "our identity provider dashboard",
            "the single sign-on gateway",
            "the corporate login flow",
        ],
        "frictions": [
            {
                "symptoms": [
                    "users are getting bounced back to the login page every 15 minutes even while actively working",
                    "sessions keep expiring mid-workday and everyone has to re-authenticate constantly",
                    "the idle timeout logs people out after just a quarter hour of use",
                ],
                "details": [
                    "It seemed to get worse after we raised the org policy strictness.",
                    "Helpdesk tickets about this spiked to ~40 last week alone.",
                    "It repros on Chrome and Safari alike.",
                ],
                "impacts": [
                    "people are losing unsaved work and the helpdesk is drowning",
                    "we're getting compliance questions because users start parking credentials in notes apps",
                    "leadership noticed the login churn in the weekly metrics review",
                ],
                "problem": "A {role} reports that active SSO sessions expire after roughly 15 minutes of use, forcing repeated re-authentication during the workday. The frequent interruptions are increasing helpdesk volume and undermining confidence in the platform.",
                "want": "a configurable idle-session timeout with a silent token-refresh flow",
                "so_that": "authenticated users stay signed in during active work without weakening security policy",
                "scenario": "SSO session expires during active use",
                "given": "a user is authenticated via SSO and has been active within the configured idle window",
                "when": "the idle timeout elapses while the user is still interacting with the application",
                "then": "the session is refreshed transparently and the user is not redirected to the identity provider",
                "ands": [
                    "no unsaved user work is lost during the refresh",
                    "the refresh events are recorded in the authentication audit log",
                ],
            },
            {
                "symptoms": [
                    "SAML logins fail with 'Assertion not yet valid' for a chunk of our users",
                    "after the clock change a bunch of SSO attempts got rejected on assertion validity",
                    "we see intermittent SAML rejections that look timestamp-related",
                ],
                "details": [
                    "The IdP and SP clocks differ by about three minutes.",
                    "Started right after the DST switch.",
                    "Affects maybe 5% of login attempts, never the same users twice.",
                ],
                "impacts": [
                    "those users are effectively locked out until we manually nudge clocks",
                    "support is walking users through workaround logins which wastes hours",
                    "our on-call keeps getting paged for it",
                ],
                "problem": "A {role} reports that SAML assertions are rejected with 'not yet valid' errors whenever identity-provider and service-provider clocks drift by a few minutes. The rejections intermittently lock users out of the application.",
                "want": "a configurable clock-skew tolerance applied when validating SAML assertions",
                "so_that": "minor clock drift between the identity provider and the service provider does not block authentication",
                "scenario": "SAML assertion rejected due to clock skew",
                "given": "the identity provider clock differs from the service provider clock by less than the configured skew tolerance",
                "when": "the service provider validates an incoming SAML assertion",
                "then": "the assertion is accepted and the user completes single sign-on",
                "ands": [
                    "the effective skew at validation time is written to the authentication log",
                    "skew beyond the tolerance still fails closed with a clear error",
                ],
            },
            {
                "symptoms": [
                    "offboarded employees still show app access for several hours after we remove them in the IdP",
                    "SCIM deprovisioning events sit queued and the accounts keep working",
                    "terminated accounts retain their session and permissions way too long",
                ],
                "details": [
                    "We verified the SCIM token is valid and groups sync fine.",
                    "Our offboarding SLA says 30 minutes; reality is closer to 6 hours.",
                    "It came up in a customer security review last month.",
                ],
                "impacts": [
                    "it's a real audit finding waiting to happen",
                    "security flagged it as an unacceptable access-lifecycle gap",
                    "we temporarily disable users manually, which nobody trusts",
                ],
                "problem": "A {role} reports that deprovisioned users retain application access for several hours after removal in the identity provider. The lag creates an access-lifecycle gap that fails security review.",
                "want": "immediate revocation of sessions and permissions when a SCIM deprovision event arrives",
                "so_that": "offboarded or suspended accounts lose access within the documented offboarding SLA",
                "scenario": "Deprovisioned user retains access",
                "given": "a user is removed or suspended in the identity provider",
                "when": "the SCIM deprovisioning event is delivered to the application",
                "then": "the user's active sessions are terminated and application permissions are revoked within the SLA window",
                "ands": [
                    "the revocation is recorded with a timestamp in the audit log",
                    "any later login attempt by that user is rejected",
                ],
            },
            {
                "symptoms": [
                    "role changes we make in Okta groups don't show up in the app until the next day",
                    "group-to-role mapping only updates on the nightly sync",
                    "promoted users keep their old permissions until some cache expires",
                ],
                "details": [
                    "Force-syncing the connector manually fixes it, so it's a scheduling issue.",
                    "It's bitten us twice during quarter-end role rotations.",
                    "The mapping UI even shows the new group, permissions just lag.",
                ],
                "impacts": [
                    "managers grant temporary elevated access as a workaround, which is worse",
                    "new team leads can't actually approve anything on day one",
                    "access reviews keep finding stale entitlements",
                ],
                "problem": "A {role} reports that role changes made in identity-provider groups do not propagate to application roles until a delayed sync completes. Users operate with stale permissions, which forces risky temporary-access workarounds.",
                "want": "event-driven synchronization of identity-provider group changes to application roles",
                "so_that": "permission changes take effect within minutes instead of after a nightly batch",
                "scenario": "Role mapping does not reflect IdP group changes",
                "given": "a user's group membership changes in the identity provider",
                "when": "the change event is received by the application",
                "then": "the user's application roles are updated to match the new group mapping within 5 minutes",
                "ands": [
                    "both the old and new role assignments are captured in the audit trail",
                    "a failed sync emits an alert to the administrators",
                ],
            },
            {
                "symptoms": [
                    "the 'remember this device' MFA option never sticks, users re-approve push notifications constantly",
                    "device trust resets every time someone closes their laptop lid",
                    "people are getting MFA fatigue from repeated push prompts on trusted devices",
                ],
                "details": [
                    "Cookie inspection shows the trust cookie expiring after the browser session.",
                    "Ticket volume about MFA prompts doubled this month.",
                    "Affects our sales team on the road the most.",
                ],
                "impacts": [
                    "users are starting to blind-approve prompts, which defeats the point of MFA",
                    "field teams lose 10+ minutes a day re-authenticating",
                    "we're worried about push-fatigue attacks on top of the annoyance",
                ],
                "problem": "A {role} reports that trusted-device status for MFA is not persisted, so users must re-approve push notifications on every new session. The repeated prompts are causing MFA fatigue and eroding the control's effectiveness.",
                "want": "persistent trusted-device tokens for MFA with an administrator-configurable lifetime",
                "so_that": "users on verified devices are not prompted for MFA on every session",
                "scenario": "MFA device trust not persisted across sessions",
                "given": "a user completes MFA and opts to trust the current device",
                "when": "the user starts a new session from that device within the trust lifetime",
                "then": "the session proceeds without an additional MFA prompt",
                "ands": [
                    "administrators can revoke all trusted devices for a user",
                    "trust expires automatically at the configured lifetime",
                ],
            },
            {
                "symptoms": [
                    "a subset of users hits an endless redirect loop between our app and the IdP on Safari",
                    "mobile webview logins bounce back and forth until the browser gives up",
                    "iOS users specifically can't get past the login redirect",
                ],
                "details": [
                    "Clearing cookies fixes it for one login, then it returns.",
                    "User-agent sniffing suggests it's all WebKit-based clients.",
                    "Console shows repeated SAML posts with growing cookie jars.",
                ],
                "impacts": [
                    "mobile users are effectively locked out",
                    "our mobile-adjacent traffic converted at half the rate last week",
                    "executives noticed because the mobile app links into the web console",
                ],
                "problem": "A {role} reports that authentication on WebKit-based browsers and mobile webviews enters a redirect loop between the application and the identity provider. Affected users cannot sign in from mobile devices at all.",
                "want": "a login flow that completes correctly on third-party-cookie-restricted browsers and webviews",
                "so_that": "mobile and Safari users can authenticate without getting stuck in a redirect loop",
                "scenario": "Authentication redirect loop on mobile webview",
                "given": "a user authenticates from a browser that blocks third-party cookies",
                "when": "the identity provider redirects the user back to the application",
                "then": "the session is established without further redirects and the user reaches the app",
                "ands": [
                    "a loop counter aborts with a actionable error after 5 redirects",
                    "the fallback flow works in embedded webviews used by the mobile app",
                ],
            },
            {
                "symptoms": [
                    "password reset emails are taking 20+ minutes to arrive, sometimes never",
                    "the 'forgot password' flow sends links that land after they've expired",
                    "reset emails intermittently vanish into spam or arrive too late",
                ],
                "details": [
                    "Our mail provider dashboards show deferred queues at peak times.",
                    "The links expire after 15 minutes, so late mail equals dead mail.",
                    "Roughly 8% of reset requests get a 'link expired' bounce-back.",
                ],
                "impacts": [
                    "locked-out users sit idle and call the helpdesk instead",
                    "account-lockout tickets are our number one category now",
                    "new-hire day-one access keeps slipping",
                ],
                "problem": "A {role} reports that password-reset emails are delayed past the link expiry window, so the delivered links are already invalid. Locked-out users idle until they escalate to the helpdesk.",
                "want": "prompt password-reset email delivery with single-use links and a reliable resend flow",
                "so_that": "users can self-service password recovery without waiting on the helpdesk",
                "scenario": "Password reset email delayed past link expiry",
                "given": "a user requests a password reset",
                "when": "the reset email is dispatched",
                "then": "the message is delivered within 2 minutes and the link remains valid for at least 30 minutes",
                "ands": [
                    "each reset link is single-use and invalidated after the password changes",
                    "requesting a new link invalidates all previously issued links",
                ],
            },
            {
                "symptoms": [
                    "SSO broke silently this week and nobody told us until users complained",
                    "the IdP rotated their signing certificate and our metadata went stale without any alert",
                    "login failures spiked after a cert renewal we weren't notified about",
                ],
                "details": [
                    "The metadata XML in our config still references the old fingerprint.",
                    "We only noticed because the login success dashboard dipped.",
                    "Sunday night maintenance window on the IdP side seems correlated.",
                ],
                "impacts": [
                    "a whole morning of failed logins before we diagnosed it",
                    "customer-facing teams lost hours fielding 'is it down' questions",
                    "our status page never even flickered, which eroded trust",
                ],
                "problem": "A {role} reports that identity-provider certificate rotations silently invalidate the service provider's stored SAML metadata, breaking SSO without any warning. Users experience an outage before administrators can react.",
                "want": "automatic metadata refresh plus proactive alerts when SAML metadata or certificates change",
                "so_that": "certificate rotations on the identity provider do not cause unplanned login outages",
                "scenario": "Certificate rotation breaks SAML metadata",
                "given": "the service provider has cached identity-provider metadata",
                "when": "the identity provider rotates its signing certificate",
                "then": "the application refreshes the metadata automatically and SSO continues without interruption",
                "ands": [
                    "administrators receive an alert if refresh fails for more than 1 hour",
                    "the previous certificate is trusted during a 48-hour overlap window",
                ],
            },
        ],
    },
    {
        "code": "billing",
        "name": "Billing / Invoicing",
        "roles": [
            "accounts payable coordinator at a manufacturing distributor",
            "finance operations manager at a design agency",
            "controller at a 300-employee services firm",
            "finance lead at a B2B SaaS startup",
            "billing support specialist at an e-commerce platform",
            "procurement analyst at a university system",
        ],
        "systems": [
            "the invoicing console",
            "the billing portal",
            "our accounts receivable module",
            "the finance dashboard",
            "the customer billing workspace",
        ],
        "frictions": [
            {
                "symptoms": [
                    "exporting more than about 5,000 invoices to CSV always fails with a 504",
                    "the invoice export button just spins and dies on bigger months",
                    "any CSV export over a few thousand rows times out on us",
                ],
                "details": [
                    "Smaller date ranges work fine, which is our current (painful) workaround.",
                    "Chrome devtools shows the request dying at exactly 60 seconds.",
                    "Month-end close requires the full-quarter export, so this is not optional.",
                ],
                "impacts": [
                    "month-end close is now a two-day manual exercise",
                    "we're stitching together partial exports in a spreadsheet, which is error-prone",
                    "our auditors asked why the reconciliations don't tie out",
                ],
                "problem": "A {role} reports that CSV exports of more than roughly 5,000 invoices fail with a gateway timeout, so large billing periods cannot be exported at all. Month-end close now depends on stitching partial manual exports.",
                "want": "a reliable bulk invoice export that completes for arbitrary date ranges",
                "so_that": "month-end close can be completed from a single export without manual stitching",
                "scenario": "Large invoice export times out",
                "given": "the billing account contains more than 5,000 invoices in the selected date range",
                "when": "the user requests a CSV export of that range",
                "then": "the export job completes successfully and the download contains every matching invoice",
                "ands": [
                    "export progress is visible while the job runs",
                    "the generated file includes a header row and one row per invoice with no truncation",
                ],
            },
            {
                "symptoms": [
                    "retrying a failed payment creates a brand new duplicate invoice",
                    "we keep finding twin invoices when a card charge is attempted twice",
                    "the retry button generates a second invoice instead of reusing the original",
                ],
                "details": [
                    "Happens most often on flaky-connection mobile checkouts.",
                    "Payment gateway webhooks firing twice seems to trigger it.",
                    "We refunded 14 duplicates last month alone.",
                ],
                "impacts": [
                    "customers get double-charged and open disputes",
                    "reconciliation with the gateway statement takes forever",
                    "we've issued apology credits and it's hitting our margin line",
                ],
                "problem": "A {role} reports that retrying a failed payment creates a duplicate invoice instead of reusing the original, resulting in double charges and customer disputes. Refunding duplicates is manual and reconciliation-heavy.",
                "want": "idempotent payment retries that reuse the original invoice",
                "so_that": "customers are never double-billed when a payment is retried",
                "scenario": "Payment retry creates duplicate invoice",
                "given": "a payment attempt on an existing invoice fails",
                "when": "the payment is retried by the customer or by the dunning system",
                "then": "the retry is applied to the same invoice and no duplicate invoice is created",
                "ands": [
                    "the invoice payment history records each attempt with its outcome",
                    "identical retry requests arriving concurrently produce exactly one charge",
                ],
            },
            {
                "symptoms": [
                    "EU customers are getting invoices with no VAT line at all",
                    "the tax field is empty on every invoice for our German and French accounts",
                    "VAT just isn't being computed for European customers",
                ],
                "details": [
                    "US invoices look correct, it's only EU billing profiles.",
                    "Our tax consultant flagged it during the quarterly review.",
                    "The customer's country is present on the invoice, tax rate just shows 0.",
                ],
                "impacts": [
                    "we're exposed on compliance and may owe back-taxes",
                    "EU customers are refusing payment until corrected invoices arrive",
                    "finance had to file amended returns",
                ],
                "problem": "A {role} reports that invoices issued to EU customers omit VAT entirely, printing a zero tax rate even when the customer country is known. The omission creates tax-compliance exposure and payment disputes.",
                "want": "correct VAT computation on invoices based on the customer's billing country",
                "so_that": "every EU invoice includes the applicable VAT rate and complies with local tax rules",
                "scenario": "VAT missing on EU invoices",
                "given": "a customer with a billing address in an EU country is invoiced",
                "when": "the invoice is generated",
                "then": "the invoice includes the correct VAT rate and amount for that country",
                "ands": [
                    "the customer's VAT identification number appears on the invoice when provided",
                    "reverse-charge treatment is applied when the customer supplies a valid VAT ID",
                ],
            },
            {
                "symptoms": [
                    "PDF invoices keep showing the customer's old address weeks after they updated it",
                    "we updated our billing address but every invoice still prints the previous one",
                    "the invoice PDF renders a stale address even though the profile shows the new one",
                ],
                "details": [
                    "The account page shows the correct new address.",
                    "Only the generated PDF is wrong, so it smells like caching.",
                    "A customer short-paid because the PO address mismatched.",
                ],
                "impacts": [
                    "customers' accounting departments reject the invoices",
                    "payments sit in limbo while we regenerate PDFs by hand",
                    "it looks sloppy in front of enterprise clients",
                ],
                "problem": "A {role} reports that generated invoice PDFs render a stale billing address even after the customer updates their profile. Enterprise accounting teams reject the invoices, delaying payment collection.",
                "want": "invoice PDFs that always render the customer's current billing details",
                "so_that": "invoices are accepted by customer accounting systems on first submission",
                "scenario": "Invoice renders stale billing address",
                "given": "a customer updates their billing address on their account",
                "when": "a new invoice PDF is generated",
                "then": "the PDF displays the current billing address from the customer profile",
                "ands": [
                    "regenerating an existing invoice uses the address that was current at issue time",
                    "address changes are captured in the account's change history",
                ],
            },
            {
                "symptoms": [
                    "proration credits look wrong every time someone changes plans mid-cycle",
                    "the credit we get for unused days doesn't match the math on upgrade",
                    "mid-month plan changes produce suspicious proration amounts",
                ],
                "details": [
                    "Our subscription platform says 'credit applied' but the amount is about half what we compute.",
                    "Downgrades credit correctly, upgrades are the problem.",
                    "Three customers forwarded their invoices showing the discrepancy.",
                ],
                "impacts": [
                    "customers are disputing the upgrade charges",
                    "finance can't explain the line items and looks bad",
                    "we're manually issuing adjustment credits every week",
                ],
                "problem": "A {role} reports that proration credits on mid-cycle plan changes are calculated incorrectly, producing amounts that do not match the expected unused-time refund. Customers dispute the resulting charges and finance issues manual corrections weekly.",
                "want": "accurate, transparent proration calculations for mid-cycle plan changes",
                "so_that": "customers are charged and credited correctly when plans change mid-cycle",
                "scenario": "Proration credit miscalculated on plan change",
                "given": "a customer on a monthly plan changes to a different plan mid-cycle",
                "when": "the prorated charges and credits for the remainder of the cycle are computed",
                "then": "the credit equals the unused portion of the old plan and the charge equals the remaining days of the new plan",
                "ands": [
                    "the invoice includes an itemized breakdown of the proration math",
                    "the calculation is identical when recomputed for audit purposes",
                ],
            },
            {
                "symptoms": [
                    "customers are getting 'payment failed' emails for invoices they already paid",
                    "the dunning system sends overdue notices on settled invoices",
                    "we keep apologizing for collection emails sent after successful payment",
                ],
                "details": [
                    "The invoice status shows 'paid' in the UI while the reminder email goes out.",
                    "Looks like the dunning job reads a stale status cache.",
                    "It's always the same-day-payment cases.",
                ],
                "impacts": [
                    "customers think we're incompetent, some threaten to churn",
                    "support spends hours on these tickets",
                    "one account escalated to their procurement lead",
                ],
                "problem": "A {role} reports that dunning emails are sent for invoices the customer has already paid, because the collection job reads stale invoice status. Customers receive threatening notices on settled accounts and escalate to support.",
                "want": "dunning emails that are suppressed the moment an invoice is paid",
                "so_that": "customers never receive collection notices for settled invoices",
                "scenario": "Dunning notice sent for paid invoice",
                "given": "an invoice is scheduled for a dunning reminder",
                "when": "the invoice is paid before the reminder is dispatched",
                "then": "the reminder is cancelled and no dunning email is sent",
                "ands": [
                    "the suppression is logged with the payment reference",
                    "a payment arriving during email dispatch still cancels the in-flight message",
                ],
            },
            {
                "symptoms": [
                    "credit notes never make it over to QuickBooks, invoices do",
                    "our accounting sync moves invoices but silently drops credit notes",
                    "refunds show in the app but our books never see them",
                ],
                "details": [
                    "The sync log doesn't even mention the credit notes.",
                    "Month-end reconciliation is a nightmare of manual journal entries.",
                    "Our accountants noticed first during close.",
                ],
                "impacts": [
                    "the books don't match the billing system and closing takes days longer",
                    "auditors sample-matched a refund that had no accounting record",
                    "we've had to book manual adjustments every month",
                ],
                "problem": "A {role} reports that credit notes are not synchronized to the accounting system even though invoices sync correctly. The books diverge from the billing platform and month-end close requires manual journal entries.",
                "want": "credit notes to sync to the accounting system with the same reliability as invoices",
                "so_that": "the accounting ledger always reflects refunds and credits without manual entry",
                "scenario": "Credit note fails to sync to accounting",
                "given": "a credit note is issued against a previously synced invoice",
                "when": "the accounting synchronization runs",
                "then": "the credit note is created in the accounting system and linked to the original invoice",
                "ands": [
                    "a sync failure raises an alert and retries with backoff",
                    "credit notes and invoices reconcile one-to-one in the monthly report",
                ],
            },
            {
                "symptoms": [
                    "multi-currency invoices are off by a few cents versus what the bank actually charged",
                    "the invoice total in EUR never quite matches our bank's rounding",
                    "we see penny discrepancies on almost every non-USD invoice",
                ],
                "details": [
                    "Differences are small but constant, 1-3 cents each.",
                    "At our volume that's a real reconciliation gap every month.",
                    "It seems to depend on which currency is the presentation currency.",
                ],
                "impacts": [
                    "automated reconciliation fails and a person reconciles by hand",
                    "customers withhold payment over mismatched totals",
                    "our finance automation is basically useless for non-USD",
                ],
                "problem": "A {role} reports that multi-currency invoice totals differ by small amounts from bank-confirmed charges due to inconsistent rounding rules. Automated reconciliation breaks and customers dispute mismatched totals.",
                "want": "consistent banker's-rounding on currency conversion applied line-by-line and at the invoice total",
                "so_that": "invoice totals in any currency match the amounts actually captured from the payment provider",
                "scenario": "Multi-currency rounding mismatch",
                "given": "an invoice is issued in a currency different from the account's base currency",
                "when": "line items and the invoice total are converted and rounded",
                "then": "the rounded line items sum exactly to the rounded invoice total",
                "ands": [
                    "the same rounding rule is applied to refunds and credit notes",
                    "the FX rate and its timestamp are printed on the invoice",
                ],
            },
        ],
    },
    {
        "code": "ingest",
        "name": "API / Data Ingestion",
        "roles": [
            "backend engineer on the integrations team",
            "data platform engineer at a retail analytics company",
            "solutions architect at a technology partner",
            "analytics engineer supporting the growth org",
            "platform site reliability engineer",
            "third-party developer building against the public API",
        ],
        "systems": [
            "the events ingestion API",
            "the webhooks subsystem",
            "the bulk import endpoint",
            "the data pipeline's REST layer",
            "the partner-facing event stream",
        ],
        "frictions": [
            {
                "symptoms": [
                    "when our endpoint is slow we get hit with an instant retry storm and events start dropping",
                    "webhook retries come back-to-back with no backoff and our queue overflows",
                    "a brief outage on our side means lost webhooks forever",
                ],
                "details": [
                    "We counted 6 retries inside 10 seconds in the logs.",
                    "There's no Retry-After to key off of.",
                    "We lost an estimated 3% of events during the last incident.",
                ],
                "impacts": [
                    "our downstream dashboards undercount and nobody trusts them",
                    "customer records silently go stale",
                    "we built a polling fallback, doubling our API costs",
                ],
                "problem": "A {role} reports that webhook deliveries retry immediately with no exponential backoff, so a briefly slow consumer ends in dropped events. Downstream datasets silently undercount and consumers build costly polling fallbacks.",
                "want": "webhook retries with exponential backoff and a documented dead-letter policy",
                "so_that": "temporarily unavailable endpoints never cause permanent event loss",
                "scenario": "Webhook events dropped during retry storm",
                "given": "a webhook consumer endpoint is temporarily unavailable",
                "when": "delivery attempts fail",
                "then": "retries follow exponential backoff for at least 24 hours before the event is dead-lettered",
                "ands": [
                    "dead-lettered events are retrievable by the consumer for replay",
                    "the consumer can signal an exact retry time via an HTTP Retry-After header",
                ],
            },
            {
                "symptoms": [
                    "the API returns 429 but gives us no hint when to try again",
                    "rate limit responses have no Retry-After or X-RateLimit headers",
                    "we're guessing backoff timing whenever we get throttled",
                ],
                "details": [
                    "The docs mention limits but don't specify header behavior.",
                    "Our client alternates between hammering and waiting far too long.",
                    "One integration partner filed a support ticket about it too.",
                ],
                "impacts": [
                    "our sync jobs fail unpredictably and page the on-call",
                    "we're throttled more because we can't back off intelligently",
                    "integration partners rate our API poorly for operability",
                ],
                "problem": "A {role} reports that HTTP 429 responses omit rate-limit and retry-timing headers, leaving clients to guess backoff behavior. Synchronization jobs fail unpredictably and cause avoidable on-call pages.",
                "want": "standard rate-limit headers including Retry-After on every 429 response",
                "so_that": "clients can honor rate limits deterministically instead of guessing",
                "scenario": "Rate limit response omits retry-after header",
                "given": "a client exceeds its API rate limit",
                "when": "the API returns a 429 response",
                "then": "the response includes Retry-After plus quota and reset headers",
                "ands": [
                    "successful responses include remaining-quota headers for the current window",
                    "the header semantics are documented in the API reference",
                ],
            },
            {
                "symptoms": [
                    "bulk uploads over about 10MB fail with a 502 from the gateway",
                    "any sizeable import payload gets killed at the proxy layer",
                    "large batch POSTs die before reaching the service",
                ],
                "details": [
                    "Splitting into 5MB chunks works, so it's a payload-size ceiling.",
                    "The 502 HTML mentions 'request entity too large' upstream.",
                    "Nightly imports are supposed to be 40MB+.",
                ],
                "impacts": [
                    "nightly integrations are fragile and fail every few days",
                    "we shard uploads in code, which slowed development",
                    "the biggest customer's data lands late",
                ],
                "problem": "A {role} reports that bulk upload requests larger than roughly 10MB are rejected by the gateway with a 502 before reaching the ingestion service. Nightly integration jobs fragment into manual shards and deliver customer data late.",
                "want": "bulk upload support for large payloads via chunked or resumable ingestion",
                "so_that": "nightly batch imports of any practical size complete without sharding workarounds",
                "scenario": "Bulk upload fails for large payloads",
                "given": "a client prepares a bulk import larger than 10MB",
                "when": "the import is submitted",
                "then": "the upload is accepted, processed completely, and reports per-record acceptance results",
                "ands": [
                    "an interrupted upload can resume from the last acknowledged chunk",
                    "payload-size limits are documented and return a 413 with guidance when exceeded",
                ],
            },
            {
                "symptoms": [
                    "when the upstream schema drifts, fields get silently coerced to null instead of failing",
                    "type changes in source data flow straight through as nulls with no error",
                    "schema drift is invisible until someone notices empty columns downstream",
                ],
                "details": [
                    "An upstream team changed a field from number to string last sprint.",
                    "No pipeline run failed; the column was just suddenly null.",
                    "Downstream ML features trained on the nulls.",
                ],
                "impacts": [
                    "we trained a model on silently corrupted features",
                    "two weeks of reports are retrospectively wrong",
                    "trust in the warehouse took a real hit",
                ],
                "problem": "A {role} reports that schema drift in source data is silently coerced to null values instead of being rejected or flagged. Corrupted columns propagate undetected into downstream reports and models.",
                "want": "strict schema validation on ingestion with explicit drift detection and alerting",
                "so_that": "schema changes in source systems surface immediately instead of corrupting data silently",
                "scenario": "Schema drift coerces values silently",
                "given": "an inbound payload contains a field whose type differs from the registered schema",
                "when": "the payload is ingested",
                "then": "the payload is quarantined and an alert identifies the drifted field and expected type",
                "ands": [
                    "quarantined payloads can be inspected and replayed after the schema is updated",
                    "deliberate schema evolution requires an explicit versioned migration",
                ],
            },
            {
                "symptoms": [
                    "resending events with the same idempotency key still creates duplicates",
                    "our Idempotency-Key header is accepted but clearly not honored",
                    "client retries with the same key produce second records",
                ],
                "details": [
                    "Reproduced with two identical POSTs 100ms apart.",
                    "The response even echoes the key back.",
                    "It only seems enforced on one specific endpoint.",
                ],
                "impacts": [
                    "at-least-once delivery from our broker now means duplicates in the app",
                    "we built an external dedupe layer, which was weeks of work",
                    "customer counts are inflated in billing-adjacent reports",
                ],
                "problem": "A {role} reports that the ingestion API accepts Idempotency-Key headers but does not honor them, so retried requests create duplicate records. Consumers are forced to build external deduplication layers.",
                "want": "server-side enforcement of idempotency keys across all ingestion endpoints",
                "so_that": "retried or duplicated client requests never produce duplicate records",
                "scenario": "Duplicate events accepted without idempotency check",
                "given": "a client sends a request carrying an idempotency key that was already processed",
                "when": "the request is submitted again",
                "then": "the API returns the original response without creating a new record",
                "ands": [
                    "idempotency records persist for at least 24 hours",
                    "the same key reused with a different payload returns a 422 conflict",
                ],
            },
            {
                "symptoms": [
                    "long syncs break partway when the pagination cursor expires",
                    "walking a large collection fails at page 40-ish with 'cursor not found'",
                    "our full-table syncs can't finish before cursors time out",
                ],
                "details": [
                    "Cursor TTL seems to be about 5 minutes.",
                    "Restarting from the beginning just burns quota.",
                    "Big tenants simply cannot complete a sync.",
                ],
                "impacts": [
                    "the largest customers' data is permanently partial",
                    "sync workers loop all day making no progress",
                    "we're billed for API calls that produce no usable data",
                ],
                "problem": "A {role} reports that pagination cursors expire before large collections finish syncing, aborting jobs mid-traversal with 'cursor not found'. The largest tenants' datasets cannot complete a full synchronization at all.",
                "want": "durable pagination cursors that survive long-running syncs",
                "so_that": "a full collection sync of any size completes without restarting from the first page",
                "scenario": "Cursor expires during pagination",
                "given": "a client is traversing a paginated collection",
                "when": "the next page is requested after an extended interval",
                "then": "the cursor remains valid for at least 60 minutes and returns the correct next page",
                "ands": [
                    "an expired cursor returns a distinct error code with a resumption hint",
                    "cursor lifetime is documented alongside the pagination model",
                ],
            },
            {
                "symptoms": [
                    "events arrive out of order and downstream state machines get confused",
                    "created_at ordering is not preserved at all in delivery",
                    "late-arriving update events overwrite newer state",
                ],
                "details": [
                    "We see updates delivered before their creates.",
                    "Partition retries seem to reorder messages.",
                    "Our consumer now has to keep full history just in case.",
                ],
                "impacts": [
                    "entity state in our mirror is wrong until a full re-sync",
                    "we built a reordering buffer, adding latency to everything",
                    "customer-facing timelines render out of order",
                ],
                "problem": "A {role} reports that events are delivered out of source order, so update events sometimes precede their creation events. Downstream mirrors compute incorrect state and consumers add buffering latency to compensate.",
                "want": "delivery that preserves source event ordering per entity, or explicit sequencing metadata that enables correct reordering",
                "so_that": "consumers can reconstruct source-accurate entity state without full re-syncs",
                "scenario": "Out-of-order event delivery",
                "given": "a source emits sequential events for the same entity",
                "when": "the events are delivered to a consumer",
                "then": "per-entity ordering is preserved, or each event carries sequence metadata enabling reordering",
                "ands": [
                    "sequence gaps are surfaced to the consumer as detectable gaps",
                    "the ordering guarantee is documented per subscription",
                ],
            },
            {
                "symptoms": [
                    "the ingest endpoint returns 200 instantly and invalid payloads just vanish",
                    "bad records get a success response and only fail somewhere in the logs",
                    "we only learn about malformed events when a downstream report looks wrong",
                ],
                "details": [
                    "Validation happens asynchronously with no failure surface.",
                    "No error report, dead-letter queue, or callback.",
                    "We found three weeks of silently dropped records.",
                ],
                "impacts": [
                    "our SLA on data completeness is being silently violated",
                    "engineers spend days correlating logs to find drops",
                    "the customer-facing dataset is missing rows nobody can explain",
                ],
                "problem": "A {role} reports that the ingestion API acknowledges payloads with a 200 before validation, so invalid records fail invisibly with no error surface. Data-completeness SLAs are breached silently and engineers trace losses through logs by hand.",
                "want": "synchronous validation with a structured per-record error report for every rejected payload",
                "so_that": "invalid records are rejected at submission time and never disappear silently",
                "scenario": "Invalid payload accepted with success status",
                "given": "a client submits a payload that violates the ingestion schema",
                "when": "the API responds",
                "then": "the response is a 4xx with a per-record error list identifying each violation",
                "ands": [
                    "partially valid batches report accepted and rejected records separately",
                    "every rejection is queryable via an errors endpoint for 7 days",
                ],
            },
        ],
    },
    {
        "code": "rbac",
        "name": "Role-Based Access Control",
        "roles": [
            "compliance auditor at a financial services firm",
            "IT administrator at a law firm",
            "product owner for internal tooling",
            "security engineer on the platform team",
            "operations director at a franchise group",
            "engineering team lead at a scale-up",
        ],
        "systems": [
            "the permissions console",
            "the admin settings area",
            "our workspace roles system",
            "the access management module",
            "the team administration panel",
        ],
        "frictions": [
            {
                "symptoms": [
                    "a custom role we built can edit records but gets denied when trying to view them",
                    "the role has edit permission but viewing the page 403s",
                    "our analyst role can save changes on a record it cannot even open",
                ],
                "details": [
                    "Edit scope was granted; view scope doesn't exist as a toggle.",
                    "The error page is a raw 403 with no explanation.",
                    "Two analysts lost a morning to this.",
                ],
                "impacts": [
                    "we had to grant full admin, which defeats least privilege",
                    "the auditors flagged the over-broad grants",
                    "people are sharing screenshots of records instead of links",
                ],
                "problem": "A {role} reports that a custom role can be granted edit permission on a resource without view permission, making the role unusable and driving administrators to grant broader access than intended. Audits flag the resulting over-privileged roles.",
                "want": "permission scopes where edit implies view, with validation preventing broken combinations",
                "so_that": "custom roles remain usable without granting administrator-level access",
                "scenario": "Custom role allows edit without view",
                "given": "an administrator creates a custom role with edit scope on a resource",
                "when": "the role definition is saved",
                "then": "view scope is granted automatically and holders can both open and modify the resource",
                "ands": [
                    "an explicit deny on view blocks edit as well, with a warning at save time",
                    "the effective permission matrix is previewable before saving the role",
                ],
            },
            {
                "symptoms": [
                    "permission changes take up to an hour to actually apply for users",
                    "role grants sit in some propagation delay before they work",
                    "new admins see 'access denied' for 45 minutes after being promoted",
                ],
                "details": [
                    "It's clearly a cache TTL somewhere.",
                    "Users retry, assume it's broken, and open tickets.",
                    "During incidents this delay is genuinely painful.",
                ],
                "impacts": [
                    "on-call handovers to a new responder stall during incidents",
                    "we get duplicate tickets because people assume the grant failed",
                    "emergency access requests now bypass the normal flow",
                ],
                "problem": "A {role} reports that permission changes take up to an hour to propagate to active users, so newly granted access is denied for a long window. Incident response handovers stall and users open duplicate tickets.",
                "want": "permission changes that take effect for users within one minute",
                "so_that": "access grants apply promptly and users are not denied with permissions they already hold",
                "scenario": "Permission change propagation delay",
                "given": "an administrator grants a new permission to a user",
                "when": "the change is saved",
                "then": "the permission is enforced for that user within 60 seconds",
                "ands": [
                    "active sessions pick up the change without requiring re-login",
                    "permission changes are visible in the audit log with an applied timestamp",
                ],
            },
            {
                "symptoms": [
                    "people who left a team months ago still see that team's dashboards",
                    "old team memberships linger and expose data nobody meant to share",
                    "transferred employees retain visibility into their former team's workspace",
                ],
                "details": [
                    "The team roster shows them removed.",
                    "We found it during a quarterly access review.",
                    "At least four former members had lingering visibility.",
                ],
                "impacts": [
                    "a genuine data-exposure incident waiting to happen",
                    "HR flagged cross-org data visibility after a departure",
                    "our access certification process keeps failing these accounts",
                ],
                "problem": "A {role} reports that users retain access to team resources after being removed from the team, because membership-derived grants are not revoked on removal. Former members keep visibility into sensitive workspaces.",
                "want": "team membership changes to revoke all membership-derived access immediately",
                "so_that": "departing or transferring members lose access to team resources at the moment of removal",
                "scenario": "Stale team membership exposes dashboards",
                "given": "a user is removed from a team",
                "when": "the removal is processed",
                "then": "all access derived from that membership is revoked immediately",
                "ands": [
                    "explicitly assigned permissions that predate the membership remain intact",
                    "the revocation is recorded in the audit log",
                ],
            },
            {
                "symptoms": [
                    "API tokens are born with the creator's full permissions by default",
                    "creating a personal access token grants it everything I can do, no scope picker",
                    "new tokens default to owner-level rights with no restriction prompt",
                ],
                "details": [
                    "The creation dialog has a name and expiry, nothing else.",
                    "A leaked token equals a leaked account.",
                    "We have hundreds of these tokens in CI systems.",
                ],
                "impacts": [
                    "our security review rated token handling as our top risk",
                    "a leaked CI token would expose everything the developer could touch",
                    "we can't attest to least privilege for machine credentials",
                ],
                "problem": "A {role} reports that newly created API tokens inherit the creator's full permissions by default with no scope selection. A single leaked token from a CI system exposes everything its owner can access, which fails security review.",
                "want": "scoped API tokens where permissions are selected explicitly at creation",
                "so_that": "machine credentials carry only the permissions their integration requires",
                "scenario": "API token defaults to owner permissions",
                "given": "a user creates a new API token",
                "when": "the creation flow is completed",
                "then": "the token is issued with only the scopes explicitly selected during creation",
                "ands": [
                    "existing tokens can have their scopes narrowed without reissuance",
                    "administrators can enforce a policy forbidding unrestricted tokens",
                ],
            },
            {
                "symptoms": [
                    "onboarding 30 seasonal staff means clicking through role assignment 30 times",
                    "there's no way to bulk-assign roles, every account is a manual flow",
                    "we script it against the UI with Selenium, which constantly breaks",
                ],
                "details": [
                    "Each assignment takes about two minutes including confirmations.",
                    "Twice a year we onboard 40+ people in one week.",
                    "The Selenium approach already caused one mis-assignment incident.",
                ],
                "impacts": [
                    "onboarding weeks are all-hands fire drills",
                    "people start work without system access and sit idle",
                    "one bad bulk script once granted the wrong role broadly",
                ],
                "problem": "A {role} reports that role assignment supports no bulk operation, so onboarding dozens of staff requires repetitive per-account flows. Seasonal onboarding weeks become manual fire drills and idle first days.",
                "want": "bulk role assignment via CSV upload or multi-select in the admin console",
                "so_that": "large groups of users receive correct roles in a single audited operation",
                "scenario": "Bulk role assignment unavailable",
                "given": "an administrator has a list of users to assign to a role",
                "when": "the bulk assignment is submitted",
                "then": "every listed user receives the role and a summary reports successes and skipped rows",
                "ands": [
                    "the operation creates one audit entry per affected user",
                    "invalid rows are reported with reasons without aborting the rest",
                ],
            },
            {
                "symptoms": [
                    "audit logs record views and edits but not data exports",
                    "who exported the customer list is completely unanswerable",
                    "export actions leave no trace in the audit trail",
                ],
                "details": [
                    "We verified with two test exports and saw nothing in the log.",
                    "Compliance asks this exact question during every review.",
                    "Exports include PII, which makes it worse.",
                ],
                "impacts": [
                    "we cannot satisfy a basic compliance requirement",
                    "an investigation into a leaked spreadsheet went nowhere",
                    "we're risking a formal finding",
                ],
                "problem": "A {role} reports that data export actions are not recorded in the audit log, making it impossible to attribute exported files to a user. Compliance reviews cannot verify who exported records containing personal data.",
                "want": "every data export to be captured in the audit log with actor, scope, and timestamp",
                "so_that": "data movement out of the platform is attributable during compliance reviews",
                "scenario": "Audit log omits data export events",
                "given": "a user with export permission exports a dataset",
                "when": "the export completes",
                "then": "an audit event records the actor, dataset scope, row count, and timestamp",
                "ands": [
                    "audit events for exports are retained for the same period as other audit data",
                    "administrators can filter the audit log to export events only",
                ],
            },
            {
                "symptoms": [
                    "guest collaborators can invite more guests without any approval",
                    "an external guest added two more external guests to our board",
                    "there's no control over guest-to-guest invitations",
                ],
                "details": [
                    "We only noticed because an unfamiliar domain appeared in the member list.",
                    "The setting to block it doesn't exist as far as we can tell.",
                    "Our security policy says externals need explicit sponsorship.",
                ],
                "impacts": [
                    "an unsanctioned external party viewed internal content",
                    "we had to audit every workspace for unknown guests",
                    "procurement and legal both got involved",
                ],
                "problem": "A {role} reports that guest collaborators can invite additional guests without approval, letting unsanctioned external parties access internal content. The violation of the external-access policy required a full workspace audit.",
                "want": "guest invitations to require approval from a workspace administrator",
                "so_that": "external access is always explicitly sponsored by an authorized member",
                "scenario": "Guest can invite additional guests",
                "given": "a guest collaborator attempts to invite a new guest",
                "when": "the invitation is submitted",
                "then": "the invitation is held pending administrator approval instead of taking effect",
                "ands": [
                    "administrators receive a notification with the pending invitation details",
                    "an administrator can disable guest invitations entirely for a workspace",
                ],
            },
            {
                "symptoms": [
                    "deleting a role leaves its grants attached to users and resources",
                    "removed roles orphan permissions that still show up in effective-access checks",
                    "the role is gone but its grants linger everywhere",
                ],
                "details": [
                    "Effective-access reports still list the deleted role's permissions.",
                    "Recreating a role with the same name resurrects the orphaned grants.",
                    "Found during access certification.",
                ],
                "impacts": [
                    "access reports are unreliable and reviews take twice as long",
                    "a recreated role accidentally restored old permissions to a new team",
                    "we manually clean grants via support tickets",
                ],
                "problem": "A {role} reports that deleting a role leaves its grants orphaned on users and resources, corrupting effective-access reports. Recreating a role with the same name can unintentionally resurrect old permissions.",
                "want": "role deletion to cascade cleanly, revoking or reassigning all dependent grants",
                "so_that": "effective-access reports always reflect an intentional, consistent permission state",
                "scenario": "Deleting role leaves orphaned grants",
                "given": "a role with active grants to users or resources",
                "when": "the role is deleted",
                "then": "all dependent grants are revoked, or the deletion is blocked with a dependency list until reassignment",
                "ands": [
                    "the deletion action records what was revoked in the audit log",
                    "effective-access checks never reference a deleted role",
                ],
            },
        ],
    },
    {
        "code": "reports",
        "name": "Reporting / Analytics",
        "roles": [
            "growth analyst at a consumer subscription company",
            "head of customer success at a B2B SaaS company",
            "marketing operations manager",
            "data analyst on the revenue team",
            "reporting coordinator in a PMO office",
            "executive assistant preparing board materials",
        ],
        "systems": [
            "the reporting dashboard",
            "the analytics workspace",
            "our reporting suite",
            "the insights platform",
            "the scheduled reports feature",
        ],
        "frictions": [
            {
                "symptoms": [
                    "the scheduled report email arrives with zero rows while the dashboard clearly has data",
                    "our Monday morning report shows an empty table every single week",
                    "scheduled exports deliver blank datasets even when data exists",
                ],
                "details": [
                    "Running the same report manually returns full results.",
                    "The email attachment is an empty sheet with headers only.",
                    "Started after we changed the report timezone.",
                ],
                "impacts": [
                    "the exec team stopped trusting the automated numbers",
                    "we re-run and forward manually every Monday",
                    "one department went back to maintaining spreadsheets",
                ],
                "problem": "A {role} reports that scheduled report emails arrive with empty result sets even though the same report returns data when run manually. Leadership has stopped trusting the automated numbers and the team re-sends results by hand.",
                "want": "scheduled reports that execute against the same query context as interactive runs",
                "so_that": "automated email reports always contain the data visible in the dashboard",
                "scenario": "Scheduled report sends empty result set",
                "given": "a saved report returns rows when executed interactively",
                "when": "the scheduled execution runs and emails the report",
                "then": "the emailed report contains the same rows as the interactive execution",
                "ands": [
                    "a scheduled run returning zero rows when the dashboard shows data triggers an operations alert",
                    "each emailed report notes its execution timestamp and filters",
                ],
            },
            {
                "symptoms": [
                    "exporting more than 100k rows to XLSX crashes the browser tab",
                    "big exports kill the tab with 'Aw, Snap' every time",
                    "any date range over 90 days fails the workbook export",
                ],
                "details": [
                    "Memory profiler shows the tab ballooning past 2GB.",
                    "CSV export of the same range works, so we convert manually.",
                    "Quarterly business reviews need the full quarter.",
                ],
                "impacts": [
                    "quarterly reporting now takes days of manual conversion",
                    "people export tiny slices and stitch them, sometimes missing rows",
                    "the QBR deck was late because of this",
                ],
                "problem": "A {role} reports that workbook exports beyond roughly 100,000 rows or 90 days of data crash the browser tab, so large reporting periods cannot be exported directly. Quarterly reporting now depends on manual slicing and conversion.",
                "want": "a server-side export that completes reliably for large result sets",
                "so_that": "any report can be exported in full without browser crashes or manual stitching",
                "scenario": "Large export crashes workbook",
                "given": "a report whose result set exceeds 100,000 rows",
                "when": "the user requests a spreadsheet export",
                "then": "the export is generated server-side and delivered complete within a documented time bound",
                "ands": [
                    "exports split across multiple sheets respect the workbook row limit",
                    "export progress is visible and the download link is emailed when ready",
                ],
            },
            {
                "symptoms": [
                    "the main dashboard takes 30+ seconds to load for quarter-over-quarter views",
                    "any comparison across two quarters basically hangs the dashboard",
                    "loading a quarter-range view is slow enough that people assume it's broken",
                ],
                "details": [
                    "Network tab shows a single 20-second query.",
                    "Narrower ranges are fine.",
                    "It got worse after we onboarded the EMEA entities.",
                ],
                "impacts": [
                    "executives refuse to open it and ask for email summaries instead",
                    "analysts batch questions to avoid the wait",
                    "ad-hoc decision-making suffers",
                ],
                "problem": "A {role} reports that the main dashboard takes over 30 seconds to render quarter-over-quarter comparisons, driving executives and analysts away from self-service analytics. Decision-making regresses to ad-hoc email summaries.",
                "want": "dashboard queries over long date ranges to return within 5 seconds",
                "so_that": "quarter-level analysis is interactive rather than a blocking wait",
                "scenario": "Dashboard times out on long ranges",
                "given": "a dashboard view spanning a multi-month or quarter-over-quarter range",
                "when": "the view is loaded",
                "then": "first meaningful render completes within 5 seconds using pre-aggregated data",
                "ands": [
                    "widgets render progressively rather than blocking on the slowest tile",
                    "range performance is covered by an automated performance test",
                ],
            },
            {
                "symptoms": [
                    "the CSV export shows timestamps in UTC while the chart displays local time",
                    "exported data never matches the dashboard because of timezone handling",
                    "the download is 8 hours off from what the chart shows",
                ],
                "details": [
                    "We're in UTC+8, so every exported timestamp is shifted.",
                    "Analysts join exports against dashboards and everything mismatches.",
                    "The export has no timezone indicator in the header.",
                ],
                "impacts": [
                    "two teams built conflicting reports from the same data",
                    "a customer-facing number was stated wrong in a review",
                    "nobody trusts exports without manual verification",
                ],
                "problem": "A {role} reports that CSV exports render timestamps in UTC while the dashboard displays workspace-local time, so exported figures never match the charts. Teams produce conflicting reports from the same underlying data.",
                "want": "exports that honor the workspace timezone and label the timezone explicitly",
                "so_that": "exported data reconciles exactly with the dashboard it came from",
                "scenario": "Export timezone differs from dashboard",
                "given": "a workspace configured with a display timezone",
                "when": "a report is exported to CSV",
                "then": "timestamps in the file match the dashboard rendering and the timezone is stated in the header",
                "ands": [
                    "an export option allows choosing UTC explicitly when needed",
                    "the timezone applied to an export is recorded in the file metadata",
                ],
            },
            {
                "symptoms": [
                    "the retention chart labels every cohort one month too early",
                    "cohort months are shifted, showing renewals a month before they happen",
                    "our churn numbers looked great until we realized the cohorts were mislabeled",
                ],
                "details": [
                    "Comparing against billing records proves the shift.",
                    "Accounts created May 31 show up in the May cohort.",
                    "Finance caught it during renewal forecasting.",
                ],
                "impacts": [
                    "our renewal forecast was off by a month",
                    "the churn number reported to the board was wrong",
                    "we manually rebuild retention in a spreadsheet now",
                ],
                "problem": "A {role} reports that the cohort retention chart assigns accounts to the wrong month, shifting renewal labels one month early. Revenue forecasts built on the chart were materially incorrect and required board-level correction.",
                "want": "cohort labels computed from account activation dates with documented month-boundary rules",
                "so_that": "retention figures reconcile with billing records month by month",
                "scenario": "Retention cohort labels off by one month",
                "given": "accounts activated on different days within a month",
                "when": "the cohort retention chart is rendered",
                "then": "each account is assigned to the cohort defined by the documented month-boundary rule and labels match billing periods",
                "ands": [
                    "the chart's cohort definition is documented and versioned",
                    "an automated test compares cohort totals against billing aggregates",
                ],
            },
            {
                "symptoms": [
                    "shared report links open without any of the filters that were applied",
                    "the 'copy link' button drops saved filters and date ranges",
                    "recipients of shared links see unfiltered chaos",
                ],
                "details": [
                    "The URL contains a report ID but no state.",
                    "We started describing filters in the email body.",
                    "Leadership thought they were looking at the right data when they weren't.",
                ],
                "impacts": [
                    "decisions were made on unfiltered data by mistake",
                    "we waste time walking people through re-applying filters",
                    "the share feature is effectively banned internally",
                ],
                "problem": "A {role} reports that shared report links omit the saved filters and date ranges, opening unfiltered views for recipients. Decisions have been made on the wrong data slice and the sharing feature is distrusted.",
                "want": "shared links that encode the full report state including filters and date ranges",
                "so_that": "recipients of a shared link see exactly the view the sender intended",
                "scenario": "Shared link drops saved filters",
                "given": "a report view with filters and date ranges applied",
                "when": "the share link is opened by another user",
                "then": "the view renders with the same filters and date ranges applied",
                "ands": [
                    "the shared view indicates which filters are applied",
                    "recipients with permission can modify filters without altering the shared link",
                ],
            },
            {
                "symptoms": [
                    "the same metric returns different values from the dashboard and the API",
                    "metric definitions drift between the UI and the reporting API",
                    "our numbers never tie out between the dashboard pull and the API pull",
                ],
                "details": [
                    "Difference is around 4%, consistently.",
                    "Looks like the API excludes cancelled trials and the UI doesn't.",
                    "Two teams quoted different churn figures in one meeting.",
                ],
                "impacts": [
                    "every metrics review starts with an argument about which number is right",
                    "engineering time goes to reconciling rather than analysis",
                    "automated reports built on the API contradict exec dashboards",
                ],
                "problem": "A {role} reports that identical metrics return different values from the dashboard and the reporting API because their definitions have drifted. Metrics reviews stall on reconciliation disputes and automated reports contradict executive dashboards.",
                "want": "a single shared metric definition serving both the dashboard and the API",
                "so_that": "the same metric returns the same value through every consumption path",
                "scenario": "API and dashboard metrics disagree",
                "given": "the same metric and parameters queried via the dashboard and the API",
                "when": "both results are retrieved",
                "then": "the values are identical, produced by one versioned metric definition",
                "ands": [
                    "metric definitions are documented with owner and version",
                    "a contract test fails on any divergence between surfaces",
                ],
            },
            {
                "symptoms": [
                    "PDF exports cut wide tables off at the page edge and drop charts",
                    "printing a report to PDF loses the right-hand columns",
                    "the exported PDF omits every visualization and truncates tables",
                ],
                "details": [
                    "Charts render as empty boxes.",
                    "Wide tables lose the last few columns entirely.",
                    "Board decks are assembled by screenshotting instead.",
                ],
                "impacts": [
                    "board materials are now assembled manually from screenshots",
                    "one committee received a truncated financial table without noticing",
                    "preparing each deck takes hours",
                ],
                "problem": "A {role} reports that PDF exports truncate wide tables at the page boundary and omit chart visualizations entirely. Board materials must be assembled manually from screenshots, and one committee received silently truncated financial tables.",
                "want": "PDF exports that paginate wide tables and render all charts",
                "so_that": "exported PDFs are complete, presentation-ready versions of the on-screen report",
                "scenario": "PDF export truncates wide tables",
                "given": "a report containing tables wider than one printed page and embedded charts",
                "when": "the report is exported to PDF",
                "then": "wide tables paginate across pages with repeated headers and every chart renders as an image",
                "ands": [
                    "landscape orientation is applied automatically when tables exceed portrait width",
                    "the exported PDF includes page numbers and the report title",
                ],
            },
        ],
    },
]

# ---------------------------------------------------------------------------
# Messy-input composition pools (shared across domains)
# ---------------------------------------------------------------------------

OPENERS = [
    "",
    "",
    "",
    "Hi team, ",
    "Hello, ",
    "Hey folks, ",
    "Quick one for whoever owns {system}: ",
    "Raising this after another frustrating morning — ",
    "Submitting on behalf of a customer who hit this too: ",
    "Second ticket I'm filing about this: ",
]

PERSONA_LINES = [
    "I'm the {role} here and I look after {system}. ",
    "As the {role}, I spend a lot of time in {system}. ",
    "I manage {system} for our org ({role}). ",
    "Our team relies on {system} all day — I'm the {role}. ",
    "Speaking as the {role} who gets paged when {system} misbehaves: ",
]

ASKS = [
    "Can we get this looked at before the next release?",
    "This needs to be prioritized honestly.",
    "Kind of urgent at this point.",
    "Happy to provide logs if useful.",
    "Please tell me there's a fix on the roadmap.",
    "We've worked around it for now but it's not sustainable.",
    "What's the ETA on a proper fix?",
    "This is the third time this quarter it's cost us real time.",
]

SIGNOFFS = [
    "Thanks,\n{name}",
    "— {name}",
    "Regards,\n{name}",
    "Thanks,\n{name} (cc: my team)",
    "",
    "",
]

NAMES = ["Dana", "Priya", "Marcus", "Elena", "Tom W.", "Sofia", "Jordan K.", "Aisha", "Chen", "Ravi", "Marta", "Ben O."]

# ---------------------------------------------------------------------------
# Canonical output composer
# ---------------------------------------------------------------------------


def compose_output(role: str, fr: dict, rng: random.Random) -> str:
    problem = fr["problem"].format(role=role)
    n_ands = rng.choice([1, 1, 2])
    ands = fr["ands"][:n_ands]
    ac_lines = ["Scenario: " + fr["scenario"], "Given " + fr["given"], "When " + fr["when"], "Then " + fr["then"]]
    ac_lines += ["And " + a for a in ands]
    return (
        "PROBLEM STATEMENT:\n"
        f"{problem}\n\n"
        "USER STORY:\n"
        f"As a {role}, I want {fr['want']}, So that {fr['so_that']}.\n\n"
        "ACCEPTANCE CRITERIA:\n"
        + "\n".join(ac_lines)
    )


def _terminate(sentence: str) -> str:
    sentence = sentence.strip()
    return sentence if sentence.endswith((".", "!", "?")) else sentence + "."


def compose_input(role: str, system: str, fr: dict, rng: random.Random) -> str:
    opener = rng.choice(OPENERS).format(system=system)
    persona = rng.choice(PERSONA_LINES).format(role=role, system=system)
    symptom = rng.choice(fr["symptoms"])
    body = [f"{persona}Basically, {symptom}."]
    if rng.random() < 0.65:
        body.append(_terminate(rng.choice(fr["details"])))
    if rng.random() < 0.7:
        body.append(_terminate(rng.choice(fr["impacts"])))
    if rng.random() < 0.75:
        body.append(rng.choice(ASKS))
    text = " ".join(body)
    signoff = rng.choice(SIGNOFFS).format(name=rng.choice(NAMES))
    parts = [p for p in (opener.strip(), text.strip()) if p]
    out = " ".join(parts)
    if signoff:
        out = out.rstrip() + "\n\n" + signoff
    return out


# ---------------------------------------------------------------------------
# Local (deterministic) generator
# ---------------------------------------------------------------------------


def generate_local(n_total: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    per_domain = n_total // len(DOMAINS)
    remainder = n_total - per_domain * len(DOMAINS)
    samples: list[dict] = []
    seen_hashes: set[str] = set()

    for domain in DOMAINS:
        count = per_domain + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        made = 0
        attempts = 0
        while made < count and attempts < count * 60:
            attempts += 1
            fr = rng.choice(domain["frictions"])
            role = rng.choice(domain["roles"])
            system = rng.choice(domain["systems"])
            inp = compose_input(role, system, fr, rng)
            h = hashlib.sha1(inp.encode()).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            made += 1
            samples.append(
                {
                    "id": f"{domain['code']}-{made:03d}",
                    "domain": domain["code"],
                    "input": inp,
                    "output": compose_output(role, fr, rng),
                }
            )
        if made < count:
            raise RuntimeError(f"could not generate {count} unique samples for {domain['code']} (got {made})")
    return samples


# ---------------------------------------------------------------------------
# Teacher-API generator (DeepSeek or any OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------

TEACHER_INSTRUCTIONS = """You are generating a synthetic dataset for fine-tuning a small model that \
converts raw user feedback into structured requirements.

Domain: {domain_name}
Friction archetype: {scenario}
Persona to write from: {role}

Produce a JSON object with exactly two keys:
- "input": messy, first-person user feedback (a support ticket or internal message) that describes \
this problem from the persona's perspective. 50-120 words. Casual, concrete, may include invented \
error messages, timestamps, or business impact. No structure.
- "output": the strict canonical transformation of that feedback, following EXACTLY this format:

PROBLEM STATEMENT:
<1-2 sentences identifying the role, the friction, and the business impact.>

USER STORY:
As a <role>, I want <capability>, So that <outcome>.

ACCEPTANCE CRITERIA:
Scenario: <name>
Given <precondition>
When <trigger>
Then <primary expected outcome>
And <secondary outcome>

Rules for "output": no greetings, no explanations, no markdown, no bullet points, \
nothing before PROBLEM STATEMENT: or after the last Gherkin line. Return only the JSON object."""


def generate_api(n_total: int, out_path: Path) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("api source requires the openai package: pip install openai")

    import os

    api_key = os.environ.get("TEACHER_API_KEY")
    if not api_key:
        sys.exit("TEACHER_API_KEY is required for --source api")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("TEACHER_BASE_URL", "https://api.deepseek.com/v1"),
    )
    model = os.environ.get("TEACHER_MODEL", "deepseek-chat")

    rng = random.Random(0)
    samples: list[dict] = []
    seen: set[str] = set()
    quota = {d["code"]: 0 for d in DOMAINS}
    per_domain = -(-n_total // len(DOMAINS))  # ceil

    for domain in DOMAINS:
        idx = 0
        while quota[domain["code"]] < per_domain and idx < per_domain * 5:
            fr = domain["frictions"][idx % len(domain["frictions"])]
            role = rng.choice(domain["roles"])
            idx += 1
            prompt = TEACHER_INSTRUCTIONS.format(
                domain_name=domain["name"], scenario=fr["scenario"], role=role
            )
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            try:
                obj = json.loads(raw)
                inp, outp = obj["input"].strip(), obj["output"].strip()
            except (json.JSONDecodeError, KeyError):
                continue
            if inp in seen:
                continue
            seen.add(inp)
            quota[domain["code"]] += 1
            samples.append(
                {
                    "id": f"{domain['code']}-{quota[domain['code']]:03d}",
                    "domain": domain["code"],
                    "input": inp,
                    "output": outp,
                }
            )
    return samples


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["local", "api"], default="local")
    ap.add_argument("--n", type=int, default=250, help="total samples (evenly split across 5 domains)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "raw" / "generated.jsonl")
    args = ap.parse_args()

    if args.n < len(DOMAINS):
        sys.exit(f"--n must be at least {len(DOMAINS)}")

    if args.source == "local":
        samples = generate_local(args.n, args.seed)
    else:
        samples = generate_api(args.n, args.out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    by_domain: dict[str, int] = {}
    for s in samples:
        by_domain[s["domain"]] = by_domain.get(s["domain"], 0) + 1
    print(f"wrote {len(samples)} samples -> {args.out}")
    for code, count in sorted(by_domain.items()):
        print(f"  {code:8s} {count}")
    print("next: python data/lint_dataset.py --in " + str(args.out))


if __name__ == "__main__":
    main()

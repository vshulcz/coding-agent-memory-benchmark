#!/usr/bin/env python3
"""Create the Wikidata item for deja-vu using an authenticated BotPassword."""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any

WIKIDATA_API = "https://www.wikidata.org/w/api.php"


def build_claim_item(prop: str, qid: str, ref_url: str | None = None) -> dict[str, Any]:
    """Construct one wikibase-entityid claim with an optional reference URL."""
    claim: dict[str, Any] = {
        "mainsnak": {
            "snaktype": "value",
            "property": prop,
            "datavalue": {
                "value": {"entity-type": "item", "numeric-id": int(qid[1:])},
                "type": "wikibase-entityid",
            },
        },
        "type": "statement",
        "rank": "normal",
    }
    if ref_url:
        claim["references"] = [
            {
                "snaks": {
                    "P854": [
                        {
                            "snaktype": "value",
                            "property": "P854",
                            "datavalue": {"value": ref_url, "type": "string"},
                        }
                    ]
                }
            }
        ]
    return claim


def build_claim_string(
    prop: str,
    value: str,
    ref_url: str | None = None,
    qualifier_item: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Construct one string/URL/external-id claim with optional reference and qualifier."""
    claim: dict[str, Any] = {
        "mainsnak": {
            "snaktype": "value",
            "property": prop,
            "datavalue": {"value": value, "type": "string"},
        },
        "type": "statement",
        "rank": "normal",
    }
    if qualifier_item:
        qprop, qid = qualifier_item
        claim["qualifiers"] = {
            qprop: [
                {
                    "snaktype": "value",
                    "property": qprop,
                    "datavalue": {
                        "value": {"entity-type": "item", "numeric-id": int(qid[1:])},
                        "type": "wikibase-entityid",
                    },
                }
            ]
        }
    if ref_url:
        claim["references"] = [
            {
                "snaks": {
                    "P854": [
                        {
                            "snaktype": "value",
                            "property": "P854",
                            "datavalue": {"value": ref_url, "type": "string"},
                        }
                    ]
                }
            }
        ]
    return claim


def build_entity_payload(zenodo_doi: str = "", zenodo_id: str = "") -> dict[str, Any]:
    """Build the complete Wikidata item payload for deja-vu."""
    repo = "https://github.com/vshulcz/deja-vu"
    claims: dict[str, list[dict[str, Any]]] = {
        "P31": [
            build_claim_item("P31", "Q341", repo),
            build_claim_item("P31", "Q115214659", repo),
            build_claim_item("P31", "Q506883", repo),
        ],
        "P275": [build_claim_item("P275", "Q334661", f"{repo}/blob/main/LICENSE")],
        "P6216": [build_claim_item("P6216", "Q50423863", f"{repo}/blob/main/LICENSE")],
        "P277": [build_claim_item("P277", "Q37227", f"{repo}/blob/main/go.mod")],
        "P306": [
            build_claim_item("P306", "Q388", f"{repo}/releases"),
            build_claim_item("P306", "Q14116", f"{repo}/releases"),
            build_claim_item("P306", "Q1406", f"{repo}/releases"),
        ],
        "P1324": [build_claim_string("P1324", repo, qualifier_item=("P8423", "Q186055"))],
        "P856": [build_claim_string("P856", "https://vshulcz.github.io/deja-vu/")],
        "P2078": [build_claim_string("P2078", "https://vshulcz.github.io/deja-vu/guide/getting-started.html")],
        "P348": [build_claim_string("P348", "0.24.2", f"{repo}/releases/tag/v0.24.2")],
        "P8262": [build_claim_string("P8262", "@vshulcz/deja-vu", "https://www.npmjs.com/package/@vshulcz/deja-vu")],
        "P8443": [build_claim_string("P8443", "deja-vu", "https://formulae.brew.sh/formula/deja-vu")],
        "P12025": [build_claim_string("P12025", "vshulcz.deja-vu", f"{repo}/tree/main/packaging/winget")],
    }
    if zenodo_doi:
        claims["P356"] = [build_claim_string("P356", zenodo_doi.upper(), "https://github.com/vshulcz/coding-agent-memory-benchmark")]
    if zenodo_id:
        claims["P4901"] = [build_claim_string("P4901", zenodo_id, "https://github.com/vshulcz/coding-agent-memory-benchmark")]

    return {
        "labels": {
            "en": {"language": "en", "value": "deja-vu"},
            "ru": {"language": "ru", "value": "deja-vu"},
        },
        "descriptions": {
            "en": {
                "language": "en",
                "value": "local cross-agent session memory and transcript search CLI for AI coding agents",
            },
            "ru": {
                "language": "ru",
                "value": "локальный CLI-инструмент поиска по истории сессий и общей памяти для ИИ-агентов программирования",
            },
        },
        "aliases": {
            "en": [{"language": "en", "value": "deja"}],
            "ru": [{"language": "ru", "value": "deja"}],
        },
        "claims": claims,
    }


def api_call(opener: urllib.request.OpenerDirector, params: dict[str, str], post: bool = False) -> dict[str, Any]:
    """Execute one MediaWiki API request against wikidata.org."""
    ua = {"User-Agent": "deja-vu-wikidata-bot/1.0 (https://github.com/vshulcz/deja-vu)"}
    if post:
        encoded = urllib.parse.urlencode(params).encode("utf-8")
        req = urllib.request.Request(WIKIDATA_API, data=encoded, headers=ua, method="POST")
    else:
        url = f"{WIKIDATA_API}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=ua, method="GET")
    with opener.open(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def create_item(user: str, password: str, payload: dict[str, Any]) -> str:
    """Authenticate via BotPassword and create a new entity with assert=user."""
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    tok = api_call(opener, {"action": "query", "meta": "tokens", "type": "login", "format": "json"})
    login_token = tok["query"]["tokens"]["logintoken"]

    login_res = api_call(
        opener,
        {
            "action": "login",
            "lgname": user,
            "lgpassword": password,
            "lgtoken": login_token,
            "format": "json",
        },
        post=True,
    )
    if login_res.get("login", {}).get("result") != "Success":
        raise RuntimeError(f"Wikidata login failed: {login_res}")

    csrf_res = api_call(opener, {"action": "query", "meta": "tokens", "type": "csrf", "format": "json"})
    csrf_token = csrf_res["query"]["tokens"]["csrftoken"]

    edit_res = api_call(
        opener,
        {
            "action": "wbeditentity",
            "new": "item",
            "data": json.dumps(payload),
            "summary": "Create deja-vu software entity (MIT, Go, coding-agent memory CLI)",
            "assert": "user",
            "token": csrf_token,
            "format": "json",
        },
        post=True,
    )
    if edit_res.get("success") != 1:
        raise RuntimeError(f"wbeditentity failed: {edit_res}")
    return edit_res["entity"]["id"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create Wikidata item for deja-vu (authenticated only).")
    parser.add_argument("--zenodo-doi", default="", help="Optional minted Zenodo DOI for benchmark archive (P356)")
    parser.add_argument("--zenodo-id", default="", help="Optional Zenodo record ID (P4901)")
    parser.add_argument("--dry-run", action="store_true", help="Print validated JSON payload without calling Wikidata")
    args = parser.parse_args(argv)

    payload = build_entity_payload(zenodo_doi=args.zenodo_doi, zenodo_id=args.zenodo_id)
    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    user = os.environ.get("WIKIDATA_USER", "").strip()
    password = os.environ.get("WIKIDATA_PASS", "").strip()
    if not user or not password:
        print("missing WIKIDATA_USER / WIKIDATA_PASS BotPassword env vars (refusing anonymous edit)", file=sys.stderr)
        return 2

    qid = create_item(user, password, payload)
    print(f"created https://www.wikidata.org/wiki/{qid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

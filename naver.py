import httpx
import re
import asyncio
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from typing import Optional

BASE = "https://openapi.naver.com/v1/search/{type}.json"

TYPE_MAP = {"news": "news", "blog": "blog", "web": "webkr", "cafe": "cafearticle"}


def _headers(client_id: str, client_secret: str) -> dict:
    return {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&[a-zA-Z#\d]+;", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_date(item: dict, stype: str) -> Optional[date]:
    """날짜 파싱 — 여러 필드를 순차적으로 시도."""
    # 1) postdate (blog, cafe: YYYYMMDD)
    try:
        d = item.get("postdate", "")
        if d and len(d) == 8 and d.isdigit():
            return date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    except Exception:
        pass

    # 2) pubDate (news, web: RFC 2822)
    try:
        pub = item.get("pubDate", "")
        if pub:
            return parsedate_to_datetime(pub).date()
    except Exception:
        pass

    # 3) 기타 형식 (YYYY-MM-DD, YYYY.MM.DD 등)
    try:
        for field in ("pubDate", "postdate", "date"):
            val = (item.get(field) or "").strip()
            if not val:
                continue
            for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d"):
                try:
                    return datetime.strptime(val[:10], fmt).date()
                except ValueError:
                    continue
    except Exception:
        pass

    return None


async def search_once(
    client_id: str,
    client_secret: str,
    query: str,
    stype: str = "news",
    display: int = 5,
    start: int = 1,
    sort: str = "date",
) -> dict:
    url = BASE.format(type=TYPE_MAP.get(stype, "news"))
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(
            url,
            headers=_headers(client_id, client_secret),
            params={"query": query, "display": min(display, 100),
                    "start": start, "sort": sort},
        )
        r.raise_for_status()
    return r.json()


async def fetch_all(
    client_id: str,
    client_secret: str,
    query: str,
    stype: str = "news",
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    max_items: int = 500,
) -> list[dict]:
    results: list[dict] = []
    start = 1

    while start <= 1000 and len(results) < max_items:
        batch = await search_once(client_id, client_secret, query, stype,
                                  display=100, start=start, sort="date")
        items = batch.get("items", [])
        if not items:
            break

        stop = False
        for item in items:
            item["title"] = _clean(item.get("title", ""))
            item["description"] = _clean(item.get("description", ""))
            d = _parse_date(item, stype)
            item["parsed_date"] = d.isoformat() if d else None

            if date_from and date_to and stype != "cafe":
                # 카페 API는 날짜 필드 없음 → 날짜 필터 스킵
                if d is None:
                    # 날짜 파싱 불가 → 범위 판단 못하므로 일단 포함
                    results.append(item)
                    continue
                if d < date_from:
                    # 최신순 정렬이므로 이 이후는 전부 범위 밖 → 수집 중단
                    stop = True
                    break
                if d <= date_to:
                    results.append(item)
                # d > date_to: 아직 범위 전 → continue (스킵하되 중단하지 않음)
            else:
                results.append(item)

        if stop or len(items) < 100:
            break
        start += 100
        await asyncio.sleep(0.1)

    return results[:max_items]

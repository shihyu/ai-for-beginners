#!/usr/bin/env python3
"""
爬取 iT邦幫忙鐵人賽系列文章並轉成 mdbook 格式
Series: 新手也能懂得AI-深入淺出的AI課程
URL: https://ithelp.ithome.com.tw/users/20152236/ironman/5607
"""

import os
import re
import time
import urllib.parse
import hashlib
from pathlib import Path

import requests
from bs4 import BeautifulSoup
import markdownify

BASE_URL = "https://ithelp.ithome.com.tw"
SERIES_URL = "https://ithelp.ithome.com.tw/users/20152236/ironman/5607"
PROJECT_DIR = Path(__file__).parent.parent
SRC_DIR = PROJECT_DIR / "src"
IMAGES_DIR = SRC_DIR / "images"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

session = requests.Session()
session.headers.update(HEADERS)


def fetch(url: str, retries: int = 3) -> BeautifulSoup:
    for i in range(retries):
        try:
            r = session.get(url, timeout=20)
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except Exception as e:
            print(f"  [retry {i+1}/{retries}] {url}: {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to fetch {url}")


def get_all_article_urls() -> list[dict]:
    """從系列頁面蒐集所有文章 URL（自動翻頁）"""
    articles = []
    page = 1

    while True:
        url = f"{SERIES_URL}?page={page}"
        print(f"[index page {page}] {url}")
        soup = fetch(url)

        # 文章連結：href 含 /articles/
        links = soup.find_all("a", href=lambda h: h and "/articles/" in h)
        if not links:
            break

        for a in links:
            href = a.get("href", "").strip()
            title = a.get_text(strip=True)
            # 找同容器內的 day 數字
            container = a.parent
            day_num = ""
            for _ in range(6):
                day_el = container.select_one(".ir-qa-list__days")
                if day_el:
                    day_num = day_el.get_text(strip=True).replace("DAY", "").strip()
                    break
                if container.parent:
                    container = container.parent
                else:
                    break

            articles.append({
                "day": day_num,
                "title": title,
                "url": href.strip(),
            })

        # 翻頁：找下一頁連結
        next_link = soup.select_one(f'a[href*="page={page+1}"]')
        if not next_link:
            break
        page += 1
        time.sleep(0.5)

    return articles


def download_image(img_url: str) -> str | None:
    """下載圖片到 images/ 目錄，返回相對路徑"""
    if not img_url or img_url.startswith("data:"):
        return None

    # 補全 URL
    if img_url.startswith("//"):
        img_url = "https:" + img_url
    elif img_url.startswith("/"):
        img_url = BASE_URL + img_url

    # 只下載 ithome 的圖片
    if "ithome.com.tw" not in img_url:
        return img_url  # 保留外部圖片原始 URL

    # 建立檔名
    parsed = urllib.parse.urlparse(img_url)
    filename = os.path.basename(parsed.path)
    if not filename or "." not in filename:
        filename = hashlib.md5(img_url.encode()).hexdigest()[:12] + ".jpg"

    dest = IMAGES_DIR / filename
    if dest.exists():
        return f"images/{filename}"

    try:
        r = session.get(img_url, timeout=20, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"    [img] ✓ {filename}")
        return f"images/{filename}"
    except Exception as e:
        print(f"    [img-err] {img_url}: {e}")
        return img_url  # fallback 保留原始 URL


def process_content(soup: BeautifulSoup) -> str:
    """提取文章主體、下載圖片、轉換 Markdown"""
    # 優先選擇器
    content = (
        soup.select_one(".markdown__style")
        or soup.select_one(".qa-markdown")
        or soup.select_one(".markdown")
    )
    if not content:
        print("    [warn] 找不到文章內容區塊")
        return ""

    # 移除廣告、社交按鈕等雜訊
    for sel in ["script", "style", ".fb-like", ".social-like", '[class*="ad-"]']:
        for el in content.select(sel):
            el.decompose()

    # 處理所有圖片：下載並替換 src
    for img in content.find_all("img"):
        src = (img.get("src") or img.get("data-src") or "").strip()
        if not src:
            img.decompose()
            continue

        new_src = download_image(src)
        if new_src:
            img["src"] = new_src
        img["alt"] = img.get("alt") or ""
        # 移除多餘 attributes
        for attr in ["width", "height", "loading", "data-src"]:
            if attr in img.attrs:
                del img.attrs[attr]

    # 轉換 HTML → Markdown
    md = markdownify.markdownify(
        str(content),
        heading_style="ATX",
        bullets="-",
        strip=["script", "style"],
    )

    # 清理：移除多餘空行、修正圖片路徑前後空格
    md = re.sub(r"\n{3,}", "\n\n", md)
    md = re.sub(r"!\[([^\]]*)\]\(\s+", r"![\1](", md)
    md = re.sub(r"\s+\)", ")", md)
    md = md.strip()
    return md


def scrape_article(url: str) -> dict:
    """爬取單篇文章"""
    soup = fetch(url)

    # 標題
    title_el = (
        soup.select_one(".qa-header h2")
        or soup.select_one("h2.qa-header__title")
        or soup.select_one("h2")
    )
    title = title_el.get_text(strip=True) if title_el else "無標題"

    # 發布日期
    date_el = soup.select_one(".qa-header__info-time") or soup.select_one("time")
    date = date_el.get_text(strip=True) if date_el else ""

    # 內容
    content_md = process_content(soup)

    return {
        "title": title,
        "date": date,
        "url": url,
        "content_md": content_md,
    }


def main():
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    SRC_DIR.mkdir(exist_ok=True)

    print("=" * 50)
    print("Step 1: 蒐集文章列表")
    print("=" * 50)
    articles = get_all_article_urls()
    print(f"\n找到 {len(articles)} 篇文章:")
    for a in articles:
        print(f"  Day {a['day']:>2}: {a['title'][:60]}")

    if not articles:
        print("ERROR: 沒有找到任何文章！請檢查 URL 或網路連線。")
        return

    print(f"\n{'=' * 50}")
    print("Step 2: 爬取每篇文章內容")
    print("=" * 50)

    results = []
    for i, meta in enumerate(articles, 1):
        day_num = meta.get("day") or str(i)
        print(f"\n[{i:2d}/{len(articles)}] Day {day_num}: {meta['title'][:50]}")
        try:
            data = scrape_article(meta["url"])
            data["day_num"] = i
            data["day"] = day_num
            if not data["title"] or data["title"] == "無標題":
                data["title"] = meta["title"]
            results.append(data)
            time.sleep(0.8)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({
                "day_num": i,
                "day": day_num,
                "title": meta["title"],
                "date": "",
                "url": meta["url"],
                "content_md": f"> ⚠️ 無法爬取此文章: {e}\n\n原始連結: [{meta['url']}]({meta['url']})",
            })

    print(f"\n{'=' * 50}")
    print("Step 3: 產生 Markdown 檔案")
    print("=" * 50)

    filenames = []
    for data in results:
        n = data["day_num"]
        fname = f"day{n:02d}.md"
        filepath = SRC_DIR / fname
        title = data["title"]

        front = f"# {title}\n\n"
        meta_lines = []
        if data.get("date"):
            meta_lines.append(f"**發布日期:** {data['date']}")
        if data.get("url"):
            meta_lines.append(f"**原文連結:** [{data['url']}]({data['url']})")
        if meta_lines:
            front += "> " + "  \n> ".join(meta_lines) + "\n\n"
        front += "---\n\n"

        filepath.write_text(front + data["content_md"] + "\n", encoding="utf-8")
        filenames.append((n, fname, title))
        print(f"  ✓ {fname}: {title[:55]}")

    print(f"\n{'=' * 50}")
    print("Step 4: 產生 SUMMARY.md")
    print("=" * 50)

    lines = [
        "# 新手也能懂得AI-深入淺出的AI課程",
        "",
        "[課程簡介](README.md)",
        "",
    ]
    for n, fname, title in filenames:
        lines.append(f"- [Day {n:02d}: {title}]({fname})")

    (SRC_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("  ✓ SUMMARY.md")

    print(f"\n{'=' * 50}")
    print("Step 5: 產生 README.md")
    print("=" * 50)

    toc = "\n".join(f"- **Day {n:02d}**: [{t}]({f})" for n, f, t in filenames)
    readme = f"""# 新手也能懂得AI-深入淺出的AI課程

> **2022 iThome 鐵人賽** | AI & Data 組 | 作者: [austin70915](https://ithelp.ithome.com.tw/users/20152236)
>
> 原始系列: <https://ithelp.ithome.com.tw/users/20152236/ironman/5607>

## 簡介

讓沒接觸過 Python 的人也能了解 AI 內部的構造與內容。

從基礎的安裝程式、安裝函式庫、基礎語法，
到 DNN、CNN、LSTM 等神經網路架構與資料前處理的技術，
最後是預訓練模型的介紹 (NLP 與 CV 兩個方向)。

## 課程大綱

{toc}
"""
    (SRC_DIR / "README.md").write_text(readme, encoding="utf-8")
    print("  ✓ README.md")

    img_count = len(list(IMAGES_DIR.glob("*")))
    print(f"\n{'=' * 50}")
    print(f"完成！共 {len(results)} 篇文章，{img_count} 張圖片")
    print(f"輸出目錄: {SRC_DIR}")
    print("=" * 50)
    print("\n下一步: make build")


if __name__ == "__main__":
    main()

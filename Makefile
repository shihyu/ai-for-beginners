.DEFAULT_GOAL := help

PYTHON := python3
MDBOOK := /home/shihyu/.mybin/mdbook
OUT_DIR := book

.PHONY: help
help:
	@echo "可用目標："
	@echo "  make scrape  - 爬取所有文章並產生 markdown"
	@echo "  make build   - 用 mdbook 編譯成 HTML"
	@echo "  make serve   - 啟動 mdbook 本地預覽 (port 3000)"
	@echo "  make all     - scrape + build"
	@echo "  make clean   - 清除 book/ 輸出目錄"
	@echo ""
	@echo "使用範例："
	@echo "  make all        # 一鍵爬取並編譯"
	@echo "  make serve      # 本地預覽"

.PHONY: scrape
scrape:
	@echo "=== 爬取文章 ==="
	$(PYTHON) scripts/scraper.py

.PHONY: build
build:
	@echo "=== 編譯 mdbook ==="
	$(MDBOOK) build

.PHONY: serve
serve:
	@echo "=== 啟動預覽 http://localhost:3000 ==="
	@lsof -ti:3000 | xargs -r kill -9 2>/dev/null || true
	$(MDBOOK) serve --port 3000

.PHONY: all
all: scrape build

.PHONY: clean
clean:
	rm -rf $(OUT_DIR)
	@echo "清除完畢"

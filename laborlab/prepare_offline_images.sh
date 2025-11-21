#!/bin/bash
# 폐쇠망용 Docker 이미지 준비 스크립트

echo "=== 폐쇠망용 Docker 이미지 준비 ==="

# 1. 베이스 이미지 다운로드
echo "1. 베이스 이미지 다운로드 중..."
docker pull python:3.11-slim

# 2. 베이스 이미지 저장
echo "2. 베이스 이미지 저장 중..."
docker save python:3.11-slim | gzip > python-3.11-slim.tar.gz

echo "3. 완료! python-3.11-slim.tar.gz 파일을 폐쇠망으로 전송하세요."
echo ""
echo "폐쇠망에서 실행할 명령어:"
echo "  docker load < python-3.11-slim.tar.gz"
echo "  docker build -f laborlab/Dockerfile -t dowhy-laborlab:latest ."


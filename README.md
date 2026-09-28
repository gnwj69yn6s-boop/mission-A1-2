import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from openai import OpenAI


# 환경변수 불러오기
load_dotenv()

RESULTS_DIR = Path("results")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
KAKAO_REST_API_KEY = os.getenv("KAKAO_REST_API_KEY")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

KAKAO_URL = "https://dapi.kakao.com/v2/local/search/keyword.json"


def parse_arguments():
    """CLI 인자를 설정하고 날짜를 반환한다."""

    parser = argparse.ArgumentParser(
        description="LLM과 Kakao Local API를 활용한 국내 여행 추천 프로그램"
    )

    parser.add_argument(
        "-date",
        "--date",
        required=True,
        help="여행 날짜를 YYYY-MM-DD 형식으로 입력하세요."
    )

    args = parser.parse_args()

    return args.date


def validate_date(date_text):
    """입력받은 날짜가 YYYY-MM-DD 형식인지 확인한다."""

    try:
        parsed_date = datetime.strptime(date_text, "%Y-%m-%d")
        return parsed_date.strftime("%Y-%m-%d")

    except ValueError:
        print("오류: 날짜 형식이 올바르지 않습니다.")
        print("사용법: python travel_planner.py --date \"2026-10-03\"")
        sys.exit(1)


def check_api_keys():
    """필수 API 키가 설정되어 있는지 확인한다."""

    missing_keys = []

    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")

    if not KAKAO_REST_API_KEY:
        missing_keys.append("KAKAO_REST_API_KEY")

    if missing_keys:
        print("\n[오류] API 키가 설정되지 않았습니다.")
        print("다음 환경변수를 설정해주세요:")

        for key in missing_keys:
            print(f"- {key}")

        print("\n.env 파일 또는 환경변수를 확인해주세요.")
        sys.exit(1)


def get_openai_client():
    """OpenAI 클라이언트를 생성한다."""

    return OpenAI(api_key=OPENAI_API_KEY)


def create_recommendation_schema():
    """1차 여행 추천 JSON Schema를 반환한다."""

    return {
        "type": "object",
        "properties": {
            "recommended_city": {
                "type": "string"
            },
            "weather": {
                "type": "string"
            },
            "events": {
                "type": "array",
                "items": {
                    "type": "string"
                }
            },
            "reason": {
                "type": "string"
            }
        },
        "required": [
            "recommended_city",
            "weather",
            "events",
            "reason"
        ],
        "additionalProperties": False
    }


def request_recommendation(client, travel_date):
    """
    여행 날짜를 기반으로 LLM에게 여행 지역을 추천받는다.

    JSON Schema를 사용하여 구조화된 JSON 응답을 요청한다.
    """

    prompt = f"""
당신은 국내 여행 전문가입니다.

사용자가 여행하려는 날짜는 {travel_date}입니다.

해당 시기에 국내에서 여행하기 좋은 지역을 하나 추천해주세요.

다음 조건을 반드시 지켜주세요.

1. recommended_city에는 국내 도시 이름을 문자열로 작성합니다.
2. weather에는 해당 시기의 일반적인 날씨 특징을 설명합니다.
3. events에는 해당 시기에 방문을 고려할 수 있는 행사나 축제 후보를
   1~3개 문자열 배열로 작성합니다.
4. reason에는 해당 지역을 추천하는 이유를 2~4문장으로 작성합니다.
5. 실제 실시간 날씨나 행사 일정이 확정되었다고 단정하지 말고,
   일반적인 계절 정보와 후보 수준으로 설명합니다.
"""

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "travel_recommendation",
                "description": "국내 여행 지역 추천 결과",
                "schema": create_recommendation_schema(),
                "strict": True
            }
        }
    )

    return response.output_text


def parse_recommendation_json(raw_text):
    """LLM 응답을 JSON으로 변환하고 필수 키를 확인한다."""

    data = json.loads(raw_text)

    required_keys = [
        "recommended_city",
        "weather",
        "events",
        "reason"
    ]

    for key in required_keys:
        if key not in data:
            raise ValueError(f"필수 키가 없습니다: {key}")

    if not isinstance(data["recommended_city"], str):
        raise ValueError("recommended_city는 문자열이어야 합니다.")

    if not isinstance(data["weather"], str):
        raise ValueError("weather는 문자열이어야 합니다.")

    if not isinstance(data["events"], list):
        raise ValueError("events는 배열이어야 합니다.")

    if not isinstance(data["reason"], str):
        raise ValueError("reason은 문자열이어야 합니다.")

    return data


def get_recommendation_with_retry(client, travel_date, errors):
    """
    여행 추천을 요청한다.

    JSON 파싱 실패 시 최대 1회 재시도한다.
    """

    print("[1/3] 1차 추천 생성 중(LLM)...")

    for attempt in range(2):
        try:
            raw_text = request_recommendation(
                client,
                travel_date
            )

            recommendation = parse_recommendation_json(raw_text)

            print(
                f"  - recommended_city: "
                f"{recommendation['recommended_city']}"
            )

            return recommendation

        except Exception as error:
            errors.append({
                "step": "llm_recommendation",
                "type": "JSON_PARSE_ERROR",
                "message": str(error)
            })

            if attempt == 0:
                print("  - JSON 파싱 실패. 1회 재시도합니다.")
                continue

            print("  - 재시도에도 실패했습니다.")

    return {
        "recommended_city": "데이터 없음",
        "weather": "데이터 없음",
        "events": [],
        "reason": "LLM 추천 결과를 생성하지 못했습니다."
    }


def search_restaurants(city, errors):
    """
    Kakao Local API로 해당 도시의 맛집을 검색한다.

    API 실패 시 빈 리스트를 반환하여 다음 단계가 계속 진행되도록 한다.
    """

    print("[2/3] 맛집 검색 중(지도/장소 API)...")

    headers = {
        "Authorization": f"KakaoAK {KAKAO_REST_API_KEY}"
    }

    params = {
        "query": f"{city} 맛집",
        "size": 5
    }

    try:
        response = requests.get(
            KAKAO_URL,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code in (401, 403):
            errors.append({
                "step": "place_search",
                "type": "AUTH_ERROR",
                "message": f"HTTP {response.status_code}"
            })

            print(
                f"  - 오류: 인증 실패({response.status_code}). "
                "Kakao REST API 키를 확인하세요."
            )

            return []

        if response.status_code == 429:
            errors.append({
                "step": "place_search",
                "type": "QUOTA_ERROR",
                "message": "HTTP 429 - API 요청 한도 초과"
            })

            print("  - 오류: API 요청 한도를 초과했습니다.")

            return []

        response.raise_for_status()

        data = response.json()

        documents = data.get("documents", [])

        if not documents:
            errors.append({
                "step": "place_search",
                "type": "EMPTY_RESULT",
                "message": f"0 results for query={city} 맛집"
            })

            print("  - 검색 결과 0건")
            print("  - 맛집 섹션을 데이터 없음으로 처리합니다.")

            return []

        restaurants = []

        for item in documents:
            restaurant = {
                "name": item.get("place_name", ""),
                "address": (
                    item.get("road_address_name")
                    or item.get("address_name")
                    or ""
                ),
                "category": item.get("category_name", ""),
                "url": item.get("place_url", ""),
                "x": safe_float(item.get("x")),
                "y": safe_float(item.get("y"))
            }

            restaurants.append(restaurant)

        print(f"  - 맛집 {len(restaurants)}곳 검색 완료")

        return restaurants

    except requests.exceptions.Timeout:
        errors.append({
            "step": "place_search",
            "type": "NETWORK_ERROR",
            "message": "Kakao API 요청 시간 초과"
        })

        print("  - 오류: 네트워크 요청 시간이 초과되었습니다.")

        return []

    except requests.exceptions.RequestException as error:
        errors.append({
            "step": "place_search",
            "type": "API_ERROR",
            "message": str(error)
        })

        print("  - 오류: Kakao API 요청에 실패했습니다.")
        print("  - 맛집 섹션을 데이터 없음으로 처리합니다.")

        return []

    except json.JSONDecodeError:
        errors.append({
            "step": "place_search",
            "type": "PARSE_ERROR",
            "message": "Kakao API 응답 JSON 파싱 실패"
        })

        print("  - 오류: API 응답을 JSON으로 읽을 수 없습니다.")

        return []


def safe_float(value):
    """좌표 문자열을 숫자로 변환한다."""

    if value is None or value == "":
        return None

    try:
        return float(value)

    except (ValueError, TypeError):
        return None


def create_report_prompt(travel_date, recommendation, restaurants, errors):
    """최종 여행 리포트 생성용 프롬프트를 만든다."""

    restaurants_text = json.dumps(
        restaurants,
        ensure_ascii=False,
        indent=2
    )

    errors_text = json.dumps(
        errors,
        ensure_ascii=False,
        indent=2
    )

    recommendation_text = json.dumps(
        recommendation,
        ensure_ascii=False,
        indent=2
    )

    return f"""
당신은 국내 여행 전문 플래너입니다.

다음 데이터를 이용하여 여행 리포트를 Markdown 형식으로 작성해주세요.

여행 날짜:
{travel_date}

1차 여행 추천 JSON:
{recommendation_text}

맛집 검색 결과:
{restaurants_text}

오류 정보:
{errors_text}

반드시 다음 목차를 포함해주세요.

# {travel_date} 국내 여행 추천 리포트

## 추천 지역

추천 지역을 작성합니다.

## 추천 이유

추천 이유를 자연스럽게 요약합니다.

## 날씨 요약

제공된 날씨 정보를 바탕으로 작성합니다.

## 행사/축제

행사나 축제 후보를 목록으로 작성합니다.

## 맛집 추천

맛집 검색 결과가 있다면 최대 5곳을 목록 또는 표로 작성합니다.

맛집 결과가 0건이면 반드시 다음처럼 작성합니다.

- 데이터 없음 (장소 검색 결과 0건 또는 API 오류)

각 맛집에 이름, 주소, 카테고리, URL을 가능한 범위에서 포함합니다.

## 1일 일정 제안

오전 / 오후 / 저녁으로 나누어 하루 일정을 제안합니다.

## 오류 요약(errors)

오류가 있다면 간단하게 정리합니다.

오류가 없다면 다음처럼 작성합니다.

- 오류 없음

주의사항:

- 제공된 데이터를 우선 사용합니다.
- 실시간 정보라고 단정하지 않습니다.
- 존재하지 않는 맛집이나 행사 정보를 임의로 만들지 않습니다.
- Markdown 문서만 출력합니다.
"""


def create_final_report(
    client,
    travel_date,
    recommendation,
    restaurants,
    errors
):
    """최종 Markdown 여행 리포트를 생성한다."""

    print("[3/3] 최종 리포트 생성 중(LLM)...")

    prompt = create_report_prompt(
        travel_date,
        recommendation,
        restaurants,
        errors
    )

    try:
        response = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt
        )

        report = response.output_text

        if not report.strip():
            raise ValueError("LLM이 빈 리포트를 반환했습니다.")

        print("  - 리포트 생성 완료")

        return report

    except Exception as error:
        errors.append({
            "step": "llm_report",
            "type": "LLM_ERROR",
            "message": str(error)
        })

        print("  - 최종 리포트 생성에 실패했습니다.")

        return create_fallback_report(
            travel_date,
            recommendation,
            restaurants,
            errors
        )


def create_fallback_report(
    travel_date,
    recommendation,
    restaurants,
    errors
):
    """LLM 최종 리포트 생성 실패 시 사용할 기본 Markdown."""

    lines = []

    lines.append(f"# {travel_date} 국내 여행 추천 리포트")
    lines.append("")

    lines.append("## 추천 지역")
    lines.append(
        f"- {recommendation.get('recommended_city', '데이터 없음')}"
    )
    lines.append("")

    lines.append("## 추천 이유")
    lines.append(recommendation.get("reason", "데이터 없음"))
    lines.append("")

    lines.append("## 날씨 요약")
    lines.append(recommendation.get("weather", "데이터 없음"))
    lines.append("")

    lines.append("## 행사/축제")

    events = recommendation.get("events", [])

    if events:
        for event in events:
            lines.append(f"- {event}")
    else:
        lines.append("- 데이터 없음")

    lines.append("")

    lines.append("## 맛집 추천")

    if restaurants:
        for restaurant in restaurants:
            lines.append(
                f"- **{restaurant['name']}**"
            )
            lines.append(
                f"  - 주소: {restaurant['address']}"
            )
            lines.append(
                f"  - 카테고리: {restaurant['category']}"
            )
            lines.append(
                f"  - URL: {restaurant['url']}"
            )
    else:
        lines.append("- 데이터 없음")

    lines.append("")

    lines.append("## 1일 일정 제안")
    lines.append("- 오전: 지역 대표 관광지 방문")
    lines.append("- 오후: 주변 관광지 및 카페 방문")
    lines.append("- 저녁: 지역 맛집 방문")
    lines.append("")

    lines.append("## 오류 요약(errors)")

    if errors:
        for error in errors:
            lines.append(
                f"- [{error['type']}] "
                f"{error['message']}"
            )
    else:
        lines.append("- 오류 없음")

    return "\n".join(lines)


def save_results(
    travel_date,
    recommendation,
    restaurants,
    errors,
    report
):
    """원본 JSON과 최종 Markdown을 results 폴더에 저장한다."""

    RESULTS_DIR.mkdir(exist_ok=True)

    json_path = RESULTS_DIR / f"{travel_date}_raw.json"
    report_path = RESULTS_DIR / f"{travel_date}_travel_plan.md"

    raw_data = {
        "date": travel_date,
        "recommendation": recommendation,
        "restaurants": restaurants,
        "errors": errors
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            raw_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    return json_path, report_path


def main():
    """프로그램 전체 실행 흐름."""

    print("=" * 60)
    print("      국내 여행 AI 추천 프로그램")
    print("=" * 60)

    travel_date = parse_arguments()

    travel_date = validate_date(travel_date)

    check_api_keys()

    errors = []

    client = get_openai_client()

    recommendation = get_recommendation_with_retry(
        client,
        travel_date,
        errors
    )

    city = recommendation["recommended_city"]

    if city == "데이터 없음":
        restaurants = []
    else:
        restaurants = search_restaurants(
            city,
            errors
        )

    report = create_final_report(
        client,
        travel_date,
        recommendation,
        restaurants,
        errors
    )

    json_path, report_path = save_results(
        travel_date,
        recommendation,
        restaurants,
        errors,
        report
    )

    print("\n" + "=" * 60)
    print("프로그램 실행이 완료되었습니다.")
    print("=" * 60)

    print(f"\n원본 데이터:")
    print(f"  {json_path}")

    print(f"\n최종 여행 리포트:")
    print(f"  {report_path}")

    print("\nresults 폴더에서 결과 파일을 확인해주세요.")


if __name__ == "__main__":
    main()

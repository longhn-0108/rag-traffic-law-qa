from traffic_law_rag.rag import (
    build_search_query,
    legal_reference,
    normalize_vietnamese,
    rrf_fusion,
    RetrievedPoint,
)


def test_normalize_vietnamese():
    assert normalize_vietnamese("Đèn Đỏ") == "den do"


def test_query_expansion():
    query = build_search_query("xe máy vượt đèn đỏ")
    assert "đèn tín hiệu giao thông" in query


def test_legal_reference():
    ref = legal_reference({
        "document_type": "Nghị định",
        "document_id": "168/2024/ND-CP",
        "article": 6,
        "clause": 3,
        "point": "a",
        "heading_title": "Xử phạt",
    })
    assert "Điều 6" in ref
    assert "Khoản 3" in ref
    assert "Điểm a" in ref


def test_rrf_fusion():
    a = RetrievedPoint({"chunk_id": "a"}, 1.0)
    b = RetrievedPoint({"chunk_id": "b"}, 1.0)
    fused = rrf_fusion([a, b], [b, a], limit=2)
    assert len(fused) == 2
    assert fused[0].payload["chunk_id"] in {"a", "b"}

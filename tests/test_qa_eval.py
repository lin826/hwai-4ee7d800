from health_context.qa_eval import QACase, grade


def test_grade_value_date_and_citation():
    case = QACase("f", "lab_latest", "q", "2025-09-29", 6.31, gold_resource_id="abc")
    assert grade(case, "6.31 % on 2025-09-29 (Observation/abc)") == (True, True)
    assert grade(case, "6.31 % on Sept 29 2025") == (False, False)


def test_grade_absence_rejects_invented_dates():
    case = QACase("f", "condition_absent", "q", expect_absent=True)
    assert grade(case, "Asthma is not recorded in this record.")[0]
    assert not grade(case, "Asthma was diagnosed on 2019-01-02.")[0]

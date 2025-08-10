import re
from typing import List, Dict


def identify_key_sections(text_by_page: List[str]) -> Dict[str, List[Dict]]:
    sections = {"permanent_exclusions": [], "waiting_periods": [], "maternity_benefits": [], "general_exclusions": [], "coverage_details": []}
    section_patterns = {
        "permanent_exclusions": [r"permanent\s+exclusion", r"permanently\s+excluded", r"not\s+covered\s+under\s+any\s+circumstances", r"exclusions\s+that\s+apply\s+throughout"],
        "waiting_periods": [r"waiting\s+period", r"waiting\s+time", r"moratorium\s+period", r"initial\s+waiting"],
        "maternity_benefits": [r"maternity\s+benefit", r"maternity\s+coverage", r"pregnancy\s+related", r"childbirth\s+expenses"],
    }
    condition_patterns = {"arthritis": r"\barthritis\b", "abortion": r"\babortion\b", "hydrocele": r"\bhydrocele\b"}
    for page_num, page_text in enumerate(text_by_page, 1):
        page_text_lower = page_text.lower()
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                if re.search(pattern, page_text_lower):
                    section_info = {"page": page_num, "text": page_text, "section_type": section_name, "conditions_found": []}
                    for condition, cond_pattern in condition_patterns.items():
                        if re.search(cond_pattern, page_text_lower):
                            section_info["conditions_found"].append(condition)
                    sections[section_name].append(section_info)
                    break
    return sections


def cross_check_specific_conditions(sections: Dict[str, List[Dict]], question: str) -> Dict[str, List[Dict] | List[str] | str | None]:
    question_lower = question.lower()
    conditions_in_question = []
    if re.search(r"\barthritis\b", question_lower):
        conditions_in_question.append("arthritis")
    if re.search(r"\babortion\b", question_lower):
        conditions_in_question.append("abortion")
    if re.search(r"\bhydrocele\b", question_lower):
        conditions_in_question.append("hydrocele")
    info_type = None
    if re.search(r"exclusion|excluded|not covered", question_lower):
        info_type = "exclusions"
    elif re.search(r"waiting|period|moratorium", question_lower):
        info_type = "waiting_periods"
    elif re.search(r"maternity|pregnancy|childbirth", question_lower):
        info_type = "maternity_benefits"
    cross_check_results = {"conditions_asked": conditions_in_question, "info_type_requested": info_type, "relevant_sections": [], "specific_findings": []}
    if info_type:
        section_key = "permanent_exclusions" if info_type == "exclusions" else info_type
        if section_key in sections:
            for section in sections[section_key]:
                for condition in conditions_in_question:
                    if condition in section["conditions_found"]:
                        cross_check_results["relevant_sections"].append(section)
                        cross_check_results["specific_findings"].append(f"{condition.title()} found in {section['section_type']} on page {section['page']}")
    return cross_check_results


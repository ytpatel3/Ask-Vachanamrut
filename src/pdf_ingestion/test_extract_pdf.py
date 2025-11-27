import pytest
from extract_pdf import normalize_unicode, simple_clean_text

FI_LIGATURE = '\ufb01'   # The 'fi' ligature character
FL_LIGATURE = '\ufb02'   # The 'fl' ligature character
FRACTION_HALF = '\u00bd' # Compatibility character: Vulgar fraction one half (½)
FULL_WIDTH_A = '\uff21'  # Compatibility character: Full-width capital A
SUPERSCRIPT_TWO = '\u00b2' # Compatibility character: Superscript two (²)

def test_normalize_unicode():
    # 1. Test NFKC Compatibility (Fractions, Full-width, Superscript)
    input_nfkc = f'Price is {FRACTION_HALF} or {FULL_WIDTH_A}lmost done ({SUPERSCRIPT_TWO}).'
    expected_nfkc = 'Price is 1/2 or Almost done (2).'
    assert normalize_unicode(input_nfkc) == expected_nfkc, \
        'Assertion 1 (NFKC): Failed to convert compatibility chars (fraction/full-width/superscript).'

    # 2. Test Manual 'fi' Ligature Replacement
    input_fi = f'Final {FI_LIGATURE}nal report.'
    expected_fi = 'Final final report.'
    assert normalize_unicode(input_fi) == expected_fi, \
        "Assertion 2: Failed to replace 'fi' ligature (\\ufb01) with 'fi'."

    # 3. Test Manual 'fl' Ligature Replacement (and combined with 'fi' for robustness)
    input_fl_mixed = f'The {FL_LIGATURE}oorplan is {FI_LIGATURE}ne.'
    expected_fl_mixed = 'The floorplan is fine.'
    assert normalize_unicode(input_fl_mixed) == expected_fl_mixed, \
        "Assertion 3: Failed to replace 'fl' ligature (\\ufb02) or handle mixed ligatures."

    # 4. Test Standard Text (No Change)
    simple_text = 'This text should not change at all.'
    assert normalize_unicode(simple_text) == simple_text, \
        'Assertion 4: Standard text was incorrectly modified.'
        
    # 5. Edge Case: Empty String
    assert normalize_unicode('') == '', \
        'Assertion 5: Empty string input did not return an empty string.'

def test_simple_clean_text():
    """Test simple_clean_text normalization: line endings, spaces, and blank lines"""
    assert simple_clean_text('Line 1\r\nLine 2\rLine 3') == 'Line 1\nLine 2\nLine 3'
    assert simple_clean_text('Too  many   spaces\t\tand\ttabs') == 'Too many spaces and tabs'
    assert simple_clean_text('Paragraph 1\n\n\n\n\nParagraph 2') == 'Paragraph 1\n\nParagraph 2'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
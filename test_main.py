# tests.py
# Run `conda install pytest`
# Run `pytest test_main.py` from the speech2phone/ directory.

"""
To use pytest, the file must be named "test***.py". For example, `tests.py` or `test_pca.py`.
Test functions must be prefixed with `test_`. So `multiply()` is not a test function, but `test_numbers_3_4()` is.
We use simple assert statements for our testing. No assertThis() or assertThat().

https://docs.pytest.org/en/latest/usage.html
http://pythontesting.net/framework/pytest/pytest-introduction/#running_pytest

pytest fixtures could be useful later, but don't worry about them for now.
"""

def multiply(a, b):
    return a * b

def test_numbers_3_4():
    assert multiply(3,4) == 12

def test_strings_a_3():
    assert multiply('a',3) == 'aaa'

all:
	python test/test_metamorphosis.py

inner:
	python test/test_inner_convergence.py

clean:
	rm -r imgs *pyc *.pvd *.vtu *.png *.pdf test/.cache > /dev/null 2>&1

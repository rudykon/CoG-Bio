.PHONY: test coverage check
test:
	python -m pytest -q
coverage:
	python -m coverage run -m pytest -q
	python -m coverage report --include='clingeo/io.py,clingeo/evaluation.py,clingeo/images.py,clingeo/pipeline.py,clingeo/postprocess/coordinate_transform.py,clingeo/fusion.py,clingeo/views.py,clingeo/projection.py,clingeo/shape.py' --fail-under=90
	python -m coverage report --include='clingeo/data/dataset.py,clingeo/data/image_io.py,clingeo/data/transforms.py' --fail-under=80
check:
	python -m compileall -q clingeo scripts tests
	python -m pytest -q

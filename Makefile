
# Add a '@' to the front of the command so it doesn't
# echo onto the shell

# target: dependencies
# 	systems commands

clean:
	@:

test:
	pytest test_main.py

hello_world:
	echo Hello world!

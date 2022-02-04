import strawberryfields as sf

# create a program
prog = sf.Program(2)

with prog.context as q:
    sf.ops.S2gate(0.543) | (q[0], q[1])

# create an engine
eng = sf.Engine("gaussian")

eng.print_applied()

eng.run(prog)

eng.print_applied()

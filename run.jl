using ONNXRunTime

# Load ONNX model
sess = ONNXRunTime.load_inference("JX_NN_model.onnx")

# Inspect input/output names (use function calls, not field access)
println("Inputs: ", sess.input_names)
println("Outputs: ", sess.output_names)

# Get input/output names
input_name = first(sess.input_names)
output_name = first(sess.output_names)

# Create a single sample input
N_in = 169  # input dimension of your FC network

x = rand(Float32, 1, N_in)  # Shape: (1, 169)

# Run inference
y = sess(Dict(input_name => x))

# Extract output
ŷ = y[output_name]

println("Output size = ", size(ŷ))
println(ŷ)

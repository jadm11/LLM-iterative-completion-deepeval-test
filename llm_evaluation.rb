# Generated from gherkin-o1-preview.feature
# I had to change the code to make it work with the latest OpenAI Ruby 
# Specifically, access_token was formerly generated as api_key in the client initialization

require 'openai'
require 'logger'
require 'matrix'

# Set up logging
logger = Logger.new('llm_evaluation.log')
logger.level = Logger::DEBUG
logger.formatter = proc do |severity, datetime, progname, msg|
  "#{datetime} - #{severity} - #{msg}\n"
end

# Get the OpenAI API key from the environment variable
openai_api_key = ENV['OPENAI_API_KEY']
if openai_api_key.nil?
  puts "Please set the OPENAI_API_KEY environment variable."
  exit
end

# Initialize the OpenAI client
$openai_client = OpenAI::Client.new(access_token: openai_api_key)

# Define the fetch_response function
def fetch_response(prompt, context)
  response = $openai_client.chat(
    parameters: {
      model: 'gpt-4',
      messages: [
        { role: 'system', content: "Context: #{context}" },
        { role: 'user', content: prompt }
      ],
      max_tokens: 100
    }
  )
  content = response['choices'].first['message']['content'].strip
  content
end

# Define the semantic_similarity function
def get_embedding(text)
  response = $openai_client.embeddings(
    parameters: {
      model: 'text-embedding-ada-002',
      input: text
    }
  )
  embedding = response['data'].first['embedding']
  embedding
end

def cosine_similarity(vec1, vec2)
  dot_product = vec1.zip(vec2).map { |a, b| a * b }.reduce(:+)
  magnitude1 = Math.sqrt(vec1.map { |x| x**2 }.reduce(:+))
  magnitude2 = Math.sqrt(vec2.map { |x| x**2 }.reduce(:+))
  similarity = dot_product / (magnitude1 * magnitude2)
  similarity
end

def semantic_similarity(actual_output, expected_outputs, threshold = 0.4)
  actual_embedding = get_embedding(actual_output)
  expected_embeddings = expected_outputs.map { |output| get_embedding(output) }
  cosine_scores = expected_embeddings.map do |expected_embedding|
    cosine_similarity(actual_embedding, expected_embedding)
  end
  semantic_passed = cosine_scores.all? { |score| score >= threshold }
  [semantic_passed, cosine_scores]
end

# Define the CustomLLMEvaluation class
class CustomLLMEvaluation
  def initialize(prompt, context, threshold = 0.4, use_dynamic_responses = false)
    @prompt = prompt
    @context = context
    @threshold = threshold
    @use_dynamic_responses = use_dynamic_responses
    @actual_output = fetch_actual_output
    @expected_responses = generate_expected_responses
  end

  def fetch_actual_output
    fetch_response(@prompt, @context)
  end

  def generate_expected_responses
    if @use_dynamic_responses
      [
        fetch_response(@prompt, @context),
        fetch_response(@prompt, @context)
      ]
    else
      [
        "To get to the other side.",
        "Because its dopaminergic neurons fired synchronously across the synapses of its caudate nucleus, triggering motor contractions propelling the organism forward, to a goal predetermined by its hippocampal road mappings."
      ]
    end
  end

  def run_evaluation
    llm_test_passed = @actual_output.strip == @expected_responses[0].strip
    semantic_passed, cosine_scores = semantic_similarity(@actual_output, @expected_responses, @threshold)
    overall_passed = llm_test_passed && semantic_passed
    generate_report(overall_passed, cosine_scores)
    overall_passed
  end

  def generate_report(passed, cosine_scores)
    # Define formatting styles
    bold = "\e[1m"
    dim = "\e[2m"
    blue = "\e[34m"
    cyan = "\e[36m"
    red = "\e[31m"
    green = "\e[32m"
    reset = "\e[0m"

    # Construct the report string
    puts "#{bold}#{blue}Test Report#{reset}"
    puts "#{dim}------------------------------#{reset}"
    puts "#{cyan}Context:#{reset} #{bold}#{@context}#{reset}"
    puts "#{cyan}Dynamic Responses Enabled:#{reset} #{bold}#{@use_dynamic_responses}#{reset}"
    puts "#{cyan}Similarity Threshold:#{reset} #{bold}#{@threshold}#{reset}"
    puts "#{cyan}Cosine Scores:#{reset} #{bold}#{cosine_scores}#{reset}"
    puts "#{cyan}Input Prompt:#{reset} #{bold}#{@prompt}#{reset}"
    puts "#{cyan}Expected Responses:#{reset}"
    @expected_responses.each_with_index do |response, index|
      puts "  #{index + 1}. #{response}"
    end
    puts "#{cyan}Actual Output:#{reset} #{bold}#{@actual_output}#{reset}"
    result_text = passed ? "#{green}PASS#{reset}" : "#{red}FAIL#{reset}"
    puts "#{bold}Result:#{reset} #{result_text}"
  end
end

# Execute the evaluation script
context = "Humor"
use_dynamic_responses = true
threshold = 0.5
prompt = "Why did the chicken cross the road?"

evaluation = CustomLLMEvaluation.new(prompt, context, threshold, use_dynamic_responses)
evaluation.run_evaluation

# frozen_string_literal: true

class DataProcessor
  def initialize(data)
    @data = data
    @results = []
  end

  def process
    @data.each do |item|
      if item[:active]
        score = compute_score(item)
        @results << { name: item[:name], score: score }
      end
    end
    @results.sort_by { |r| -r[:score] }
  end

  def compute_score(item)
    item[:values].sum / item[:values].length.to_f
  rescue ZeroDivisionError => e
    puts "Error: #{e.message}"
    0.0
  end

  def risky_eval(code)
    eval(code)
  end

  def run_command(cmd)
    system(cmd)
  end
end

processor = DataProcessor.new([])
processor.process { |r| yield r }

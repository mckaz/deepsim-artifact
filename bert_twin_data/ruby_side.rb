require "rdl"
require "types/core"

NUM_DATA_POINTS = 100000 ## number of data points to generate
DIFF_TYPE_SPLIT = 0.7 ## proportion of data points that correspond to *different* types
VALUE_TYPE = "arg" ## "arg" or "ret"

DATA_FILE = "../../type-data.json"


class DataGenerator

    @vectorized_vars = {} ## Maps VarType object IDs to true/false, indicating whether or not they have been vectorized already
  
  def self.read_in_data(file_name)
    puts "Reading in data from file...".green
    return JSON.parse(File.read(file_name))
  end

  ## Given original JSON data, turn it into hash with structure
  ## { program1: { type1: [data1, data2, data3], type2: ... }, program2: ... }
  def self.restructure_data(orig_data)
    puts "Restructuring data...".green
    new_data = {}
    orig_data.each { |program, prog_data|
      new_data[program] = {}
      prog_data.each { |klass, meths|
        meths.each { |meth_name, meth_data|
          source = meth_data["source"]
          next if source.nil?
          if (VALUE_TYPE == "arg")
            next if meth_data["params"].empty?
            meth_data["params"].each { |param, pdata|
              next unless pdata["type"]
              type = pdata["type"]
              new_data[program][type] ||= [] ## will be list of [param name, method source] pairs
              new_data[program][type] << [param, source]
            }
          elsif (VALUE_TYPE == "ret")
            ret_type = meth_data["return"]["type"]
            new_data[program][ret_type] ||= [] ## will be list of source codes of methods
            new_data[program][ret_type] << source
          else
            raise "Unexpected value type #{VALUE_TYPE}."
          end
        }
      }
    }
=begin
    ## go through and delete types which don't have at least two data points
    new_data.delete_if { |prog, prog_data|
      prog_data.delete_if { |type, type_data|
        type_data.size < 2
      }
      prog_data.empty?
    }
=end
    return new_data
  end

  def self.generate_data_points(data)
    puts "Generating data points...".green
    #targets = [] ## all 0s or 1s
    points = [] ## each element has structure [input 1, input 2, target]
    0.upto(NUM_DATA_POINTS - 1).each { |n|
      target = (n < NUM_DATA_POINTS * DIFF_TYPE_SPLIT) ? 0 : 1
      rand_prog = data.keys.sample
      if n < NUM_DATA_POINTS * DIFF_TYPE_SPLIT
        until data[rand_prog].size > 1
          rand_prog = data.keys.sample ## need data for two distinct types
        end
        type1 = data[rand_prog].keys.sample
        input1 = data[rand_prog][type1].sample
        type2 = data[rand_prog].keys.reject { |t| t == type1 }.sample
        input2 = data[rand_prog][type2].sample
      else
        until data[rand_prog].any? { |type, tlist| tlist.size >= 2 }
          rand_prog = data.keys.sample
        end
        type1 = data[rand_prog].keys.sample
        until data[rand_prog][type1].size > 1
          type1 = data[rand_prog].keys.sample
        end
        input1 = data[rand_prog][type1].sample
        type2 = type1
        input2 = data[rand_prog][type2].reject { |i| i == input1 }.sample
      end
      points << [input1, input2, target] unless input1.nil? || input2.nil?
    }
    return points
  end

  def self.vectorize_data(points)
    puts "Vectorizing data...".green
    count = 1
    points.each { |in1, in2, _|
      puts "Vectorizing point ##{count}"
      if VALUE_TYPE == "arg"
        ## each input has structure [param name, source]
        vectorize_var(in1[1], in1.object_id, in1[0])
        vectorize_var(in2[1], in2.object_id, in2[0])
      elsif VALUE_TYPE == "ret"
        ## each input is the source code of a method
        vectorize_var(in1, in1.object_id)
        vectorize_var(in2, in2.object_id)
      else
        raise "Unexpected value type #{VALUE_TYPE}."
      end
      count += 1
    }
  end

  def self.save_points(points)
    puts "Saving vectorized points...".green
    count = 0
    points.each { |in1, in2, target|
      puts "Saving point #{count}..."
      count += 1
      params = { action: "save_point", in1: in1.object_id, in2: in2.object_id, target: target }
      RDL::Heuristic.send_query(params)
    }
    params = { action: "save_all_points", value_type: "#{VALUE_TYPE}" }
    RDL::Heuristic.sxend_query(params)
  end

  def self.vectorize_var(source, obj_id, param=nil)
    return true if @vectorized_vars[obj_id] ## already vectorized and cached server side
    #puts "About to vectorize #{param}"
    if VALUE_TYPE == "arg"
      raise "Expected value for param." if param.nil?
      begin
        ast = Parser::CurrentRuby.parse source
      rescue Parser::SyntaxError
        return nil
      end
      return nil if ast.nil? || !((ast.type == :def) || (ast.type == :defs))
      locs = get_var_loc(ast, param)
      if locs.empty?
        #puts "Couldn't find any locations for param #{param} in source #{source}"
        return nil
      end
      source = ast.loc.expression.source
      #puts "Querying for var #{param}"
      #puts "Sanity check: "
      #locs.each_slice(2) { |b, e| puts "    #{source[b..e]} from #{b}..#{e}" }
      params = { source: source, action: "bert_vectorize", object_id: obj_id, locs: locs, category: "arg" }
      RDL::Heuristic.send_query(params)
    elsif VALUE_TYPE == "var"
      raise "Not yet implemented."
=begin
      var_type.meths_using_var.each { |klass, meth|
        ast = RDL::Typecheck.get_ast(klass, meth)
        locs = get_var_loc(ast, var_type)
        source = ast.loc.expression.source
        puts "Querying for var #{var_type.name} in method #{klass}##{meth}"
        puts "Sanity check: "
        locs.each_slice(2) { |b, e| puts "    #{source[b..e]} from #{b}..#{e}" }
        params = { source: source, action: "bert_vectorize", object_id: var_type.object_id, locs: locs, category: "var", average: false }
        send_query(params)
      }
       send_query({ action: "bert_vectorize", object_id: var_type.object_id, category: "var", average: true })
=end
    elsif VALUE_TYPE == "ret"
      ast = Parser::CurrentRuby.parse source
      return nil if ast.nil?
      begin_pos = ast.loc.expression.begin_pos
      locs = [ast.loc.name.begin_pos - begin_pos, ast.loc.name.end_pos - begin_pos-1] ## for now, let's just try using method name
      locs = locs + RDL::Heuristic.get_ret_sites(ast)
      
      #puts "Querying for return"
      #puts "Sanity check: "
      #locs.each_slice(2) { |b, e| puts "    #{source[b..e]} from #{b}..#{e}" }
      params = { source: source, action: "bert_vectorize", object_id: obj_id, locs: locs, category: "ret" }
      RDL::Heuristic.send_query(params)
    else
      raise "not implemented yet"
    end
    @vectorized_vars[obj_id] = true
    return true
  end


  def self.get_var_loc(ast, param_name=nil)
    begin_pos = ast.loc.expression.begin_pos
    locs = []

    if ast.type == :def
      meth_name, args, body = *ast
    elsif ast.type == :defs
      _, meth_name, args, body = *ast
    else
      raise RuntimeError, "Unexpected ast type #{ast.type}"
    end

    if VALUE_TYPE == "arg"
      args.children.each { |c|
        if (c.children[0].to_s == param_name)
          ## Found the arg corresponding to var_type
          locs << (c.loc.name.begin_pos - begin_pos) ## translate it so that 0 is first position
          locs << (c.loc.name.end_pos - begin_pos - 1)
          return locs if $use_only_param_position
        end
      }
    end

    RDL::Heuristic.search_ast_for_var_locs(body, param_name, locs, begin_pos)

    raise "Expected even number of locations." unless (locs.length % 2) == 0
    #raise "Did not find var #{var_type} anywhere in ast #{ast}." if locs.length == 0
    return locs.sort ## locs will be sorted Array<Integer>, where every two ints designate the beginning/end of an occurence of the arg in var_type
    
  end

  
end

DATA = DataGenerator.read_in_data(DATA_FILE)
DATA = DataGenerator.restructure_data(DATA)
POINTS = DataGenerator.generate_data_points(DATA)
DataGenerator.vectorize_data(POINTS)
DataGenerator.save_points(POINTS)




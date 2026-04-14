local M = {}

local function compute(values)
    local sum = 0
    for i = 1, #values do
        sum = sum + values[i]
    end
    return sum / #values
end

function M.process(data)
    local results = {}
    for _, item in ipairs(data) do
        if item.active then
            local score = compute(item.values)
            table.insert(results, { name = item.name, score = score })
        end
    end
    table.sort(results, function(a, b) return a.score > b.score end)
    return results
end

function M.create_worker()
    return coroutine.create(function(data)
        for _, v in ipairs(data) do
            coroutine.yield(v * 2)
        end
    end)
end

-- dangerous
globalVar = "not local"
loadstring("print('hello')")()

local mt = {
    __index = function(t, k) return rawget(t, k) or 0 end,
    __tostring = function(t) return "MyTable" end,
}

return M

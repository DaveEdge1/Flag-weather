-- Link the first mention of each glossary term on a page to its entry on glossary.html.
-- Terms come from _glossary.yml. Headings, existing links and code are never linked.
-- Links carry class "gloss" and the definition in data-def (shown as a tooltip by glossary.html).

local terms = {}      -- { {tokens = {...}, id = ..., def = ...}, ... } longest first
local linked = {}     -- ids already linked on this page
local href_base = 'glossary.html'

local function split(s)
  local t = {}
  for w in s:gmatch('%S+') do t[#t + 1] = w end
  return t
end

local function load_terms()
  local dir = quarto.project.directory or '.'
  local f = io.open(pandoc.path.join({ dir, '_glossary.yml' }), 'r')
  if not f then return end
  local txt = f:read('a')
  f:close()
  local meta = pandoc.read('---\n' .. txt .. '\n---\n', 'markdown').meta
  for _, e in ipairs(meta.terms or {}) do
    local id, def = pandoc.utils.stringify(e.id), pandoc.utils.stringify(e.def)
    for _, a in ipairs(e.aliases or {}) do
      terms[#terms + 1] = { tokens = split(pandoc.utils.stringify(a)), id = id, def = def }
    end
  end
  table.sort(terms, function(a, b)
    if #a.tokens ~= #b.tokens then return #a.tokens > #b.tokens end
    return table.concat(a.tokens, ' '):len() > table.concat(b.tokens, ' '):len()
  end)
end

local LEAD = '^([%(%[%"\'“‘]*)(.-)$'
local TRAIL = '^(.-)([%.,;:%)%]%"\'”’!%?]*)$'

-- try to match term at position i of inlines; returns (last index, prefix, suffix) or nil
local function match_at(ils, i, term)
  local j, pre, post = i, '', ''
  for k, tok in ipairs(term.tokens) do
    local el = ils[j]
    if not el or el.t ~= 'Str' then return nil end
    local text = el.text
    if k == 1 then pre, text = text:match(LEAD) end
    if k == #term.tokens then text, post = text:match(TRAIL) end
    if text:lower() ~= tok:lower() then return nil end
    if k < #term.tokens then
      local sp = ils[j + 1]
      if not sp or (sp.t ~= 'Space' and sp.t ~= 'SoftBreak') then return nil end
      j = j + 2
    end
  end
  return j, pre, post
end

local function link_inlines(ils)
  local out = pandoc.Inlines({})
  local i = 1
  while i <= #ils do
    local done = false
    if ils[i].t == 'Str' then
      for _, term in ipairs(terms) do
        if not linked[term.id] then
          local j, pre, post = match_at(ils, i, term)
          if j then
            local content = pandoc.Inlines({})
            for k = i, j do content:insert(ils[k]:clone()) end
            local first, last = content[1], content[#content]
            first.text = first.text:sub(#pre + 1)
            last.text = last.text:sub(1, #last.text - #post)
            if pre ~= '' then out:insert(pandoc.Str(pre)) end
            out:insert(pandoc.Link(content, href_base .. '#' .. term.id, '',
              pandoc.Attr('', { 'gloss' }, { { 'data-def', term.def } })))
            if post ~= '' then out:insert(pandoc.Str(post)) end
            linked[term.id] = true
            i = j + 1
            done = true
            break
          end
        end
      end
    end
    if not done then
      out:insert(ils[i])
      i = i + 1
    end
  end
  return out
end

function Pandoc(doc)
  local input = quarto.doc.input_file or ''
  if input:match('glossary%.qmd$') then return doc end
  load_terms()
  if #terms == 0 then return doc end
  local offset = quarto.project.offset or '.'
  href_base = (offset == '.' and '' or offset .. '/') .. 'glossary.html'
  local skip = function(el) return el, false end
  return doc:walk({
    traverse = 'topdown',
    Header = skip, Link = skip, Code = skip, CodeBlock = skip, RawInline = skip, RawBlock = skip,
    Inlines = function(ils) return link_inlines(ils) end,
  })
end

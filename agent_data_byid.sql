SELECT 
public.abmem_agent.id,
public.abmem_agent.state,
public.abmem_agent.budget,
public.abmem_agent.type,
public.abmem_agent.market_id,
public.abmem_plant.id as plant,
public.abmem_plant.capacity,
public.abmem_resource.name as resource
FROM public.abmem_agent
INNER JOIN public.abmem_portfolio ON public.abmem_portfolio.agent_id =public.abmem_agent.id
INNER JOIN public.abmem_plant ON public.abmem_plant.portfolio_id = public.abmem_portfolio.id
INNER JOIN public.abmem_resource ON public.abmem_resource.id = public.abmem_plant.resource_id
WHERE public.abmem_agent.id = 187;
from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Modell példányosítása
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

# 1. City Agent
search_tool = TavilySearchResults(max_results=3)
city_agent = create_agent(
    model=llm,
    tools=[search_tool],
    system_prompt="Te egy város-specialista vagy. Találd meg a legnépszerűbb futóhelyszíneket."
)

# 2. Time Agent
@tool
def get_weather(city: str):
    """Lekéri az aktuális időjárást és előrejelzést egy adott városhoz."""
    # Itt maradt a minta válasz, de a modell tudni fogja, hogy ez a forrás
    return "16:00-kor 18°C, tiszta idő várható Kaposváron."

time_agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="Te egy időpont-tervező vagy. Az időjárás adatok alapján javasolj öltözéket."
)

# 3. Route Agent
route_agent = create_agent(
    model=llm,
    tools=[], 
    system_prompt="Te egy útvonal-tervező vagy. Feladatod konkrét távok (pl. 10km) alapján útvonalat javasolni."
)

# --- Wrapper Tool-ok az Orchestrator számára ---

@tool
def city_expert_tool(city_name: str):
    """Használd ezt az eszközt, ha városrészeket vagy helyszíneket kell keresni egy adott városban."""
    return city_agent.invoke({"messages": [("user", f"Keress futóhelyszíneket itt: {city_name}")]})

@tool
def time_expert_tool(query: str):
    """Használd ezt az eszközt az időjárás és az optimális időpont meghatározásához."""
    return time_agent.invoke({"messages": [("user", query)]})

@tool
def route_expert_tool(query: str):
    """Használd ezt az eszközt konkrét futóútvonal és távterv elkészítéséhez."""
    return route_agent.invoke({"messages": [("user", query)]})

# --- Orchestrator összeállítása ---

orchestrator_tools = [city_expert_tool, time_expert_tool, route_expert_tool]

orchestrator = create_agent(
    model=llm,
    tools=orchestrator_tools,
    system_prompt=(
        "Te vagy a futó-tervező koordinátora. A felhasználó kérésére hívd meg a CityExpert-et a helyszínekért, "
        "a TimeExpert-et az időjárásért, majd a RouteExpert-et a konkrét táv megtervezéséhez. "
        "A válaszod egy professzionális, minden részletre kiterjedő futóterv legyen."
    )
)

# --- Futtatás ---
if __name__ == "__main__":
    query = "Kaposváron, a toponári városrész közelében szeretnék futni ma 16:00-kor, kb. 10 km-t. Tervezz nekem útvonalat és nézd meg az időjárást!"
    
    # Az invoke hívás elindítja a láncot
    response = orchestrator.invoke({"messages": [("user", query)]})
    
    # Az eredmény kiírása (az utolsó üzenet az agenttől)
    print(response["messages"][-1].content)
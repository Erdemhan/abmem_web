
from services.agent import agent_factory as AgentFactory
from services.starter import starter_service
from services.logger import logger_service as LoggerService
logger = LoggerService.setupFileLogger()


def main():
    starter_service.start()

if __name__ == "__main__":
    main()
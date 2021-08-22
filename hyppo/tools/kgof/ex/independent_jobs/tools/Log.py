import logging


class Log(object):
    level_set = False
    
    @staticmethod
    def set_loglevel(loglevel):
        Log.get_logger().setLevel(loglevel)
        Log.get_logger().info("Set loglevel to %d" % loglevel)
    
    @staticmethod
    def get_logger():
        return logging.getLogger("independent-jobs") 

if not Log.level_set:
    level = logging.INFO
    logging.basicConfig(format='%(levelname)s: %(asctime)s: %(module)s.%(funcName)s(): %(message)s',
                        level=level)
    Log.get_logger().info("Global logger initialised with loglevel %d" % level)
    Log.level_set = True

logger = Log.get_logger()
class Time(object):
    @staticmethod
    def sec_to_all(seconds):
        days, seconds = divmod(seconds, 24*60*60)
        hours, seconds = divmod(seconds, 60*60)
        minutes, seconds = divmod(seconds, 60)
    
        return (days, hours, minutes, seconds)
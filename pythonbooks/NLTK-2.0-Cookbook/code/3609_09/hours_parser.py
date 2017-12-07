import datetime
from simpleparse.common import chartypes, calendar_names, iso_date_loose
from simpleparse.parser import Parser
from simpleparse.dispatchprocessor import DispatchProcessor

# chartypes provides whitespace
# calendar_names provides locale_day_names and locale_day_abbrs
# iso_date_loose provides ISO_time_loose

grammar = r'''
root	:= rule+
rule	:= ts, day, ts, '-', ts, day, ts, time, ts, '-', ts, time, ts
day		:= (locale_day_names / locale_day_names_lc / locale_day_abbrs / locale_day_abbrs_lc)
time	:= ISO_time_loose, (ts, ampm)?
ampm	:= c'a' / c'p', '.'?, ts, c'm', '.'?
ts		:= whitespace*
'''

class RecurrenceProcessor(DispatchProcessor):
	def __init__(self, *args, **kwargs):
		self.mxinterp = iso_date_loose.MxInterpreter()
		# default offset is 1 (for monday), but python's calendar module has
		# monday as 0, sunday as six
		self.dayinterp = calendar_names.DayNameInterpreter(offset=0)
	
	def ts(self, *args):
		pass
	
	def time(self, (tag, left, right, sublist), buf):
		relative_time = None
		ampm = None
		
		for item in sublist:
			if item[0] == 'ISO_time_loose':
				relative_time = self.mxinterp(item, buf)
			# ampm will always occur after ISO_time_loose, so we'll always have
			# relative_time in this case
			elif item[0] == 'ampm' and relative_time:
				start, stop = item[1:3]
				# get the AM/PM section of the string
				ampm = buf[start:stop].lower()
				# if time should be PM but isn't, adjust it
				if 'p' in ampm and relative_time.hour < 12:
					relative_time.hour += 12
		# the MxInterpreter returns mx.RelativeDateTime for ISO_time_loose
		# so we convert it to regular datetime.time
		return {'time': datetime.time(hour=relative_time.hour, minute=relative_time.minute)}
		
	def day(self, (tag, left, right, sublist), buf):
		day = None
		
		for item in sublist:
			if item[0].startswith('locale_day'):
				# this gives us a day number, where 0 is Sunday, 1 is Monday, etc
				day = self.dayinterp(item, buf)
				break
		
		return {'day': day}
	
	def rule(self, (tag, left, right, sublist), buf):
		days = []
		times = []
		
		for item in sublist:
			result = self(item, buf)
			
			if not isinstance(result, dict):
				continue
			# collect days & times
			if 'day' in result:
				days.append(result['day'])
			elif 'time' in result:
				rel = result['time']
				times.append(datetime.time(hour=rel.hour, minute=rel.minute))
		
		return {'days': days, 'times': times}

class RecurrenceParser(Parser):
	def buildProcessor(self):
		return RecurrenceProcessor()

parser = RecurrenceParser(grammar, 'root')

def parse(text):
	'''
	>>> parse('mon-tue 8A.M.-9P.M.')
	[{'days': [0, 1], 'times': [datetime.time(8, 0), datetime.time(21, 0)]}]
	>>> parse('Mon - Tue 8 A.M. - 9 P.M.')
	[{'days': [0, 1], 'times': [datetime.time(8, 0), datetime.time(21, 0)]}]
	>>> parse('Foo - Bar 9 - 11')
	>>> parse('Monday - Tuesday 08:00 - 21:00')
	[{'days': [0, 1], 'times': [datetime.time(8, 0), datetime.time(21, 0)]}]
	>>> parse('monday - tuesday 8 am - 9 pm')
	[{'days': [0, 1], 'times': [datetime.time(8, 0), datetime.time(21, 0)]}]
	>>> parse('mon-tue 8-11 wed-thu 1pm-5pm')
	[{'days': [0, 1], 'times': [datetime.time(8, 0), datetime.time(11, 0)]}, {'days': [2, 3], 'times': [datetime.time(13, 0), datetime.time(17, 0)]}]
	'''
	success, results, next = parser.parse(text)
	
	if not success or not results:
		return None
	
	return results

if __name__ == '__main__':
	import doctest
	doctest.testmod()
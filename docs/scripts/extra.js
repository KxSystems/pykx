/*
 * Common script for code.kx.com subsites
 * Author: gkelly@kx.com
 * Version: 2020.11.30
 */
 // open external link in a new tab
(function() {
	const isInternalLink = new RegExp('/' + window.location.host + '/');
	document.querySelectorAll('a').forEach(function (a) {
		if ( ! isInternalLink.test(a.href) ) {
			a.setAttribute("target", "blank");
		}
	});
})();
